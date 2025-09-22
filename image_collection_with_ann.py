# Purpose:
#   1) Export a media collection (choose MEDIA_FIELDS) -> CSV
#   2) Export one annotation set (choose ANN_FIELDS) -> CSV
#   3) Hard-filter annotations to only those media that are in the collection
#   4) (Optional) Download pixels for the media in the collection

import os, io, json
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from sqapi.api import SQAPI
from sqapi.media import SQMediaObject

# =========================
# CONFIG — EDIT THESE
# =========================
HOST             = "https://squidle.org"
TOKEN            = "591c807df9b5a7fe286626f1c3e54e5e5f50bfb1d79aa525dfb26fd4"      # put your API Token here
COLLECTION_ID    = 14233                    # <-- choose a collection id to download
ANNSET_ID        = 17711                     # <-- choose an annotation set if there's one
DOWNLOAD_IMAGES  = True                     # set False if you don't want to save images
MAX_WORKERS      = 8
OUT_DIR          = Path("out_bundle")

# Fields you want included in the collection CSV
MEDIA_FIELDS = [
    "id","key","timestamp_start","path_best","path_best_thm",
    "deployment.id","deployment.key","deployment.campaign.id","deployment.campaign.key",
    "pose.lat","pose.lon","pose.dep",
]

# Fields you want included in the annotation CSV (include ann-set + media ids for auditing)
ANN_FIELDS = [
    "id","type",
    "annotation_set_id","point.annotation_set_id",
    "label.id","label.name","label_scheme.id","label_scheme.name",
    "point.x","point.y","point.media.id",
    "created_at","user.username",
]

# =========================
# AUTH
# =========================
def get_sqapi() -> SQAPI:
    sq = SQAPI(api_key=TOKEN)
    print(f"sqapi using login {sq.current_user['username']}")
    return sq

# =========================
# EXPORT HELPERS
# =========================
def export_media_collection_csv(sqapi: SQAPI, collection_id: int,
                                include_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Export a media collection to a CSV using server-side json_normalize.
    Falls back to JSON + client-side normalize if the template isn't available.
    """
    include = list(set((include_columns or []) + ["id","path_best"]))
    # preferred path: server-side normalize → CSV
    try:
        req = sqapi.export(f"/api/media_collection/{int(collection_id)}/export",
                           include_columns=include)
        req = req.file_op("json_normalize", module="pandas").template("dataframe.csv")
        r = req.execute()  # 202 task; polled by client
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        # fallback: ask for JSON, normalize locally
        req = sqapi.export(f"/api/media_collection/{int(collection_id)}/export",
                           include_columns=include).template("json")
        js = req.execute().json()
        recs = js.get("objects", js)
        df = pd.json_normalize(recs)
        keep = [c for c in include if c in df.columns]
        return df[keep] if keep else df

def export_annotation_set_csv(sqapi: SQAPI, annset_id: int,
                              include_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Export the whole annotation set.
    """
    try:
        req = sqapi.export(f"/api/annotation_set/{int(annset_id)}/export",
                           include_columns=include_columns or [])
        req = req.file_op("json_normalize", module="pandas").template("dataframe.csv")
        r = req.execute()
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        # retry without include_columns
        req = sqapi.export(f"/api/annotation_set/{int(annset_id)}/export")
        req = req.file_op("json_normalize", module="pandas").template("dataframe.csv")
        r = req.execute()
        df = pd.read_csv(io.StringIO(r.text))
        if include_columns:
            cols = [c for c in include_columns if c in df.columns]
            if cols:
                df = df[cols]
        return df

# =========================
# JOIN / FILTER
# =========================
def detect_ann_media_col(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if c.lower().endswith(
        ("point.media.id","media.id","media_id","point_media_id")
    )]
    if cands: return cands[0]
    for c in df.columns:
        cl = c.lower()
        if "media" in cl and "id" in cl:
            return c
    raise ValueError(f"Could not find a media-id column in annotations. First cols: {list(df.columns)[:20]}")

def post_filter_ann_to_media_ids(ann_df: pd.DataFrame, media_ids: list[int]) -> pd.DataFrame:
    """Guarantee only annotations for media in this collection remain."""
    if ann_df.empty:
        return ann_df
    col = detect_ann_media_col(ann_df)
    mids = pd.to_numeric(ann_df[col], errors="coerce").astype("Int64")
    keep = mids.isin(pd.Index(media_ids, dtype="Int64"))
    return ann_df[keep].copy()

# =========================
# DOWNLOAD IMAGES (optional)
# =========================
def download_images_from_export(media_df: pd.DataFrame, out_dir: Path,
                                url_col="path_best", id_col="id",
                                max_workers: int = MAX_WORKERS) -> dict[int, Path]:
    from cv2 import imwrite, cvtColor, COLOR_RGB2BGR
    out_dir.mkdir(parents=True, exist_ok=True)
    def _one(mid: int, url: str):
        try:
            img = SQMediaObject(url, media_type="image", media_id=mid).data()  # RGB ndarray
            name = f"{mid}_{os.path.basename(urlparse(url).path)}"
            p = out_dir / name
            imwrite(str(p), cvtColor(img, COLOR_RGB2BGR))
            return mid, p
        except Exception as e:
            print(f"[warn] download failed for media {mid}: {e}")
            return mid, None

    saved: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for _, row in media_df.iterrows():
            url = str(row.get(url_col, "") or "")
            if not url: continue
            mid = int(row[id_col])
            futs[ex.submit(_one, mid, url)] = mid
        for fut in as_completed(futs):
            mid, p = fut.result()
            if p: saved[mid] = p
    return saved

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    sq = get_sqapi()

    # 1) MEDIA — authoritative list of images in this collection
    media_df = export_media_collection_csv(sq, COLLECTION_ID, MEDIA_FIELDS)
    media_csv = OUT_DIR / f"collection_{COLLECTION_ID}.csv"
    media_df.to_csv(media_csv, index=False)
    print(f"[media] rows: {len(media_df)}  -> {media_csv}")

    media_ids = media_df["id"].astype(int).tolist()

    # 2) ANNOTATIONS — export the exact set, then hard-filter to the collection’s media
    ann_df = export_annotation_set_csv(sq, ANNSET_ID, ANN_FIELDS)
    ann_df = post_filter_ann_to_media_ids(ann_df, media_ids)

    # Provenance (helps auditing later)
    ann_df["collection_id"] = COLLECTION_ID
    if "annotation_set_id" not in ann_df.columns:
        ann_df["annotation_set_id"] = ANNSET_ID  # constant if the field wasn’t returned

    ann_csv = OUT_DIR / f"annset_{ANNSET_ID}__filtered_to_collection_{COLLECTION_ID}.csv"
    ann_df.to_csv(ann_csv, index=False)
    print(f"[ann  ] rows: {len(ann_df)}  -> {ann_csv}")

    # 3) (Optional) DOWNLOAD PIXELS
    if DOWNLOAD_IMAGES:
        img_dir = OUT_DIR / "images"
        saved = download_images_from_export(media_df, img_dir, max_workers=MAX_WORKERS)
        print(f"[dl   ] downloaded {len(saved)} images to {img_dir}")
