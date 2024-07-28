import hashlib
import logging
import os
from functools import partial
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Optional, cast

import funcy
from dvc_objects.executors import ThreadPoolExecutor
from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from fsspec.utils import nullcontext

from dvc_data.callbacks import TqdmCallback
from dvc_data.hashfile.hash_info import HashInfo
from dvc_data.hashfile.state import StateBase

from .db.reference import ReferenceHashFileDB
from .hash import LargeFileHashingCallback, _hash_file, hash_file
from .meta import Meta
from .obj import HashFile

if TYPE_CHECKING:
    from typing import BinaryIO

    from dvc_objects.fs.base import AnyFSPath, FileSystem

    from ._ignore import Ignore
    from .db import HashFileDB
    from .state import StateBase
    from .tree import Tree


DefaultIgnoreFile = ".dvcignore"


class IgnoreInCollectedDirError(Exception):
    def __init__(self, ignore_file: str, ignore_dirname: str) -> None:
        super().__init__(
            f"{ignore_file} file should not be in collected dir path: "
            f"'{ignore_dirname}'"
        )


logger = logging.getLogger(__name__)


_STAGING_MEMFS_PATH = "dvc-staging"


def _upload_file(
    from_path: "AnyFSPath",
    fs: "FileSystem",
    odb: "HashFileDB",
    upload_odb: "HashFileDB",
    callback: Optional[Callback] = None,
) -> tuple[Meta, HashFile]:
    from dvc_objects.fs.utils import tmp_fname

    from .hash import HashStreamFile

    tmp_info = upload_odb.fs.join(upload_odb.path, tmp_fname())
    with fs.open(from_path, mode="rb") as stream:
        hashed_stream = HashStreamFile(stream)
        size = fs.size(from_path)
        cb = callback or TqdmCallback(
            desc=upload_odb.fs.name(from_path),
            bytes=True,
            size=size,
        )
        with cb:
            fileobj = cast("BinaryIO", hashed_stream)
            upload_odb.fs.put_file(fileobj, tmp_info, size=size, callback=cb)

    oid = hashed_stream.hash_value
    odb.add(tmp_info, upload_odb.fs, oid)
    meta = Meta(size=size)
    return meta, odb.get(oid)


def _build_file(path, fs, name, odb=None, upload_odb=None, dry_run=False):
    state = odb.state if odb else None
    meta, hash_info = hash_file(path, fs, name, state=state)
    if upload_odb and not dry_run:
        assert odb
        assert name == "md5"
        return _upload_file(path, fs, odb, upload_odb)

    oid = hash_info.value
    if dry_run:
        obj = HashFile(path, fs, hash_info)
    else:
        odb.add(path, fs, oid, hardlink=False)
        obj = odb.get(oid)

    return meta, obj


@funcy.print_durations
def _build_tree(
    path,
    fs,
    fs_info,
    name,
    odb=None,
    ignore: Optional["Ignore"] = None,
    callback: "Callback" = DEFAULT_CALLBACK,
    **kwargs,
):
    from .db import add_update_tree
    from .hash_info import HashInfo
    from .tree import Tree

    value = fs_info.get(name)
    if odb and value:
        try:
            tree = Tree.load(odb, HashInfo(name, value))
            return Meta(nfiles=len(tree)), tree
        except FileNotFoundError:
            pass

    path = path.rstrip(fs.sep)

    if ignore:
        walk_iter = ignore.walk(fs, path)
    else:
        walk_iter = fs.walk(path)

    tree_meta = Meta(size=0, nfiles=0, isdir=True)
    # assuring mypy that they are not None but integer
    assert tree_meta.size is not None
    assert tree_meta.nfiles is not None

    tree = Tree()

    upload_odb = kwargs.get("upload_odb")
    dry_run = kwargs.get("dry_run", False)
    for root, _, files in walk_iter:
        # NOTE: might happen with s3/gs/azure/etc, where empty
        # objects like `dir/` might be used to create an empty dir
        fnames: list[str] = [fname for fname in files if fname != ""]
        if not fnames:
            continue

        # NOTE: we know for sure that root starts with path, so we can use
        # faster string manipulation instead of a more robust relparts()
        rel_key: tuple[str, ...] = ()
        if root != path:
            rel_key = tuple(root[len(path) + 1 :].split(fs.sep))

        objects = _build_files(
            path,
            root,
            fnames,
            fs=fs,
            odb=odb,
            name=name,
            callback=callback,
            upload_odb=upload_odb,
            dry_run=dry_run,
        )

        for fname, (meta, hi) in objects.items():
            if fname == DefaultIgnoreFile:
                raise IgnoreInCollectedDirError(
                    DefaultIgnoreFile,
                    fs.join(root, DefaultIgnoreFile),
                )

            key: tuple[str, ...] = (*rel_key, fname)
            tree.add(key, meta, hi)
            tree_meta.size += meta.size or 0
            tree_meta.nfiles += 1

    if not tree_meta.nfiles:
        # This will raise FileNotFoundError if it is a
        # broken symlink or TreeError
        next(iter(fs.ls(path)), None)

    tree.digest()
    tree = add_update_tree(odb, tree)
    return tree_meta, tree


_url_cache: dict[str, str] = {}


def _make_staging_url(fs: "FileSystem", odb: "HashFileDB", path: Optional[str]):
    from dvc_objects.fs import Schemes

    url = f"{Schemes.MEMORY}://{_STAGING_MEMFS_PATH}-{odb.hash_name}"

    if path is not None:
        if odb.fs.protocol == Schemes.LOCAL:
            path = os.path.abspath(path)

        if path not in _url_cache:
            _url_cache[path] = hashlib.sha256(path.encode("utf-8")).hexdigest()

        url = fs.join(url, _url_cache[path])

    return url


def _get_staging(odb: "HashFileDB") -> "ReferenceHashFileDB":
    """Return an ODB that can be used for staging objects.

    Staging will be a reference ODB stored in the the global memfs.
    """

    from dvc_objects.fs import MemoryFileSystem

    fs = MemoryFileSystem()
    path = _make_staging_url(fs, odb, odb.path)
    state = odb.state
    return ReferenceHashFileDB(fs, path, state=state, hash_name=odb.hash_name)


def _build_external_tree_info(odb: "HashFileDB", tree: "Tree", name: str) -> "Tree":
    # NOTE: used only for external outputs. Initial reasoning was to be
    # able to validate .dir files right in the workspace (e.g. check s3
    # etag), but could be dropped for manual validation with regular md5,
    # that would be universal for all clouds.
    assert odb
    assert name != "md5"

    assert tree.fs
    assert tree.path
    assert tree.hash_info
    assert tree.hash_info.value

    oid = tree.hash_info.value
    odb.add(tree.path, tree.fs, oid)
    raw = odb.get(oid)
    _, hash_info = hash_file(raw.path, raw.fs, odb.hash_name, state=odb.state)

    assert hash_info.value

    tree.path = raw.path
    tree.fs = raw.fs
    tree.hash_info.name = hash_info.name
    tree.hash_info.value = hash_info.value

    if not tree.hash_info.value.endswith(".dir"):
        tree.hash_info.value += ".dir"
    return tree


def build(
    odb: "HashFileDB",
    path: "AnyFSPath",
    fs: "FileSystem",
    name: str,
    upload: bool = False,
    dry_run: bool = False,
    **kwargs,
) -> tuple["HashFileDB", "Meta", "HashFile"]:
    """Stage (prepare) objects from the given path for addition to an ODB.

    Returns at tuple of (object_store, object) where addition to the ODB can
    be completed by transferring the object from object_store to the dest ODB.

    If dry_run is True, object hashes will be computed and returned, but file
    objects themselves will not be added to the object_store ODB (i.e. the
    resulting file objects cannot transferred from object_store to another
    ODB).

    If upload is True, files will be uploaded to a temporary path on the dest
    ODB filesystem, and built objects will reference the uploaded path rather
    than the original source path.
    """
    assert path
    # assert protocol(path) == fs.protocol

    details = fs.info(path)
    staging = _get_staging(odb)

    if details["type"] == "directory":
        meta, obj = _build_tree(
            path,
            fs,
            details,
            name,
            odb=staging,
            upload_odb=odb if upload else None,
            dry_run=dry_run,
            **kwargs,
        )
        logger.debug("built tree '%s'", obj)
        if name != "md5":
            obj = _build_external_tree_info(odb, obj, name)
    else:
        meta, obj = _build_file(
            path,
            fs,
            name,
            odb=staging,
            upload_odb=odb if upload else None,
            dry_run=dry_run,
        )

    return staging, meta, obj


def _build_files(  # noqa: PLR0913
    path: str,
    root: str,
    fnames: list[str],
    fs: "FileSystem",
    name: str,
    odb: Optional["HashFileDB"] = None,
    callback: "Callback" = DEFAULT_CALLBACK,
    upload_odb: Optional["HashFileDB"] = None,
    dry_run: bool = False,
    jobs: Optional[int] = None,
    large_file_threshold: int = 2**20,
) -> dict[str, tuple[Meta, HashInfo]]:
    paths: list[str] = [f"{root}{fs.sep}{fname}" for fname in fnames]
    state: Optional[StateBase] = odb.state if odb else None

    state_data: dict[str, tuple[Meta, HashInfo, dict]] = {}
    large_files_to_hash: list[tuple[str, dict]] = []
    small_files_to_hash: list[tuple[str, dict]] = []

    infos: dict[str, dict] = {path: fs.info(path) for path in paths}
    if state and paths:
        for p, meta, hi in state.get_many(paths, fs, infos):
            if meta is not None and hi is not None and hi.name == name:
                state_data[p] = (meta, hi, infos[p])
            else:
                info = infos[p]
                size = info.get("size")
                if size and size > large_file_threshold:
                    large_files_to_hash.append((p, info))
                else:
                    small_files_to_hash.append((p, info))

    callback.set_size((callback.size or 0) + len(paths))
    callback.relative_update(len(state_data))
    hashes_to_update = _hash_files(
        fs,
        name,
        small_files_to_hash,
        large_files_to_hash,
        callback=callback,
        jobs=jobs,
    )

    if state and hashes_to_update:
        items = ((path, hi, info) for path, (_, hi, info) in hashes_to_update.items())
        state.save_many(items, fs)
        state_data.update(hashes_to_update)

    objects: dict[str, tuple[Meta, HashInfo]] = {}
    if upload_odb is not None and not dry_run:
        assert odb is not None
        assert name == "md5"
        for fname, p in zip(fnames, paths):
            meta, obj = _upload_file(p, fs, odb, upload_odb)
            objects[fname] = (meta, obj.hash_info)
        return objects

    if not dry_run:
        assert odb is not None
        to_add = {p: oid for p in paths if (oid := state_data[p][1].value) is not None}
        oids = list(to_add.values())
        paths = list(to_add)
        odb.add(paths, fs, oids, hardlink=False)

    for fname, p in zip(fnames, paths):
        meta, hi, _ = state_data[p]
        objects[fname] = meta, hi
    return objects


def _hash_single_file(
    fs: "FileSystem",
    name: str,
    arg: tuple[str, dict],
) -> tuple[str, tuple[Meta, HashInfo, dict]]:
    p, info = arg
    size = info.get("size")
    context = nullcontext(None)
    if size and size > LargeFileHashingCallback.LARGE_FILE_SIZE:
        context = LargeFileHashingCallback(desc=p)

    with context as cb:
        oid, meta = _hash_file(p, fs, name, info=info, callback=cb)
    return p, (meta, HashInfo(name, oid), info)


def _hash_files(
    fs: "FileSystem",
    name: str,
    small_files_to_hash: list[tuple[str, dict]],
    large_files_to_hash: list[tuple[str, dict]],
    callback: "Callback" = DEFAULT_CALLBACK,
    jobs: Optional[int] = None,
) -> dict[str, tuple[Meta, HashInfo, dict]]:
    if jobs is None:
        jobs = max(cpu_count(), 6)
    assert jobs is not None

    hashes: dict[str, tuple[Meta, HashInfo, dict]] = {}

    for p, info in small_files_to_hash:
        oid, meta = _hash_file(p, fs, name, info=info)
        hashes[p] = meta, HashInfo(name, oid), info
        callback.relative_update()

    if large_files_to_hash:
        func = partial(_hash_single_file, fs, name)
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            large_files = executor.imap_unordered(func, large_files_to_hash)
            hashes.update(callback.wrap(large_files))
    return hashes
