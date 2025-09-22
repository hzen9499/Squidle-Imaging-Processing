# single_media_multi_annotations.py
# Requires: pip install sqapi pandas opencv-python

import os, io
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sqapi.api import SQAPI
from sqapi.media import SQMediaObject

# =========================
# CONFIG — EDIT THESE
# =========================
HOST             = "https://squidle.org"
TOKEN            = "47d8c05612303db0996f9bc12bc0f528422e9615f81dcdc34a769c"    # or paste token below
MEDIA_ID         = 2683099                # <-- required
ANNOTATION_IDS   = [13624243, 13624245]                    # <-- [] means "all annotations on this image"; else list of IDs
ANNSET_ID        = None                   # <-- optional; if set, keep only annotations from this set
ENFORCE_ANNSET   = True                   # when ANNSET_ID is set, drop rows not matching that set
OUT_DIR          = Path("out_single_multi")
DOWNLOAD_IMAGE   = True
DRAW_OVERLAY     = True

MEDIA_FIELDS = [
    "id","key","timestamp_start","path_best","path_best_thm",
    "media_type.name","deployment.id","deployment.key",
    "deployment.campaign.id","deployment.campaign.key",
    "pose.lat","pose.lon","pose.dep",
]
ANN_FIELDS = [
    "id","type",
    "annotation_set_id","point.annotation_set_id",
    "label.id","label.name","label_scheme.id","label_scheme.name",
    "point.x","point.y","point.media.id",
    "created_at","user.username",
]
# =========================

def get_sqapi() -> SQAPI:
    sq = SQAPI(api_key=TOKEN)
    print(f"sqapi using login {sq.current_user['username']}")
    return sq

def get_by_dotted(d: dict, dotted: str, default=""):
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur

def detect_ann_media_col(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if c.lower().endswith(
        ("point.media.id","media.id","media_id","point_media_id")
    )]
    if cands: return cands[0]
    for c in df.columns:
        cl = c.lower()
        if "media" in cl and "id" in cl:
            return c
    raise ValueError(f"Could not find media-id column. First cols: {list(df.columns)[:20]}")

def export_all_annotations_for_media(sq: SQAPI, media_id: int,
                                     annset_id: int | None,
                                     ann_fields: list[str]) -> pd.DataFrame:
    """
    All annotations on this image (optionally restricted to annset).
    Uses export + server-side json_normalize; hard-filters locally by media.
    """
    import io as _io
    for media_field in ("point.media.id", "point.media_id"):
        try:
            req = sq.export("/api/annotation/export",
                            include_columns=ann_fields,
                            limit=200000, offset=0)
            if annset_id is not None:
                req = req.filter("annotation_set_id", "eq", int(annset_id))
            req = (req.filter(media_field, "eq", int(media_id))
                     .file_op("json_normalize", module="pandas")
                     .template("dataframe.csv"))
            r = req.execute()
            df = pd.read_csv(_io.StringIO(r.text))
            if not df.empty:
                col = detect_ann_media_col(df)
                df = df[pd.to_numeric(df[col], errors="coerce").astype("Int64") == int(media_id)].copy()
            return df
        except Exception:
            continue
    # fallback JSON + normalize
    for media_field in ("point.media.id", "point.media_id"):
        try:
            req = sq.export("/api/annotation/export",
                            include_columns=ann_fields,
                            limit=200000, offset=0)
            if annset_id is not None:
                req = req.filter("annotation_set_id", "eq", int(annset_id))
            r = (req.filter(media_field, "eq", int(media_id))
                   .template("json")).execute()
            js = r.json()
            recs = js.get("objects", js)
            df = pd.json_normalize(recs)
            if not df.empty:
                col = detect_ann_media_col(df)
                df = df[pd.to_numeric(df[col], errors="coerce").astype("Int64") == int(media_id)].copy()
            return df
        except Exception:
            continue
    return pd.DataFrame(columns=ann_fields)

def fetch_annotations_by_ids_for_media_via_get(sq: SQAPI, media_id: int,
                                               annotation_ids: list[int],
                                               annset_id: int | None,
                                               ann_fields: list[str]) -> pd.DataFrame:
    """
    Robust for small lists: GET /api/annotation/<id> per ID.
    - keeps only rows whose point.media.id == media_id
    - if annset_id is provided and ENFORCE_ANNSET=True, keeps only rows whose annotation_set_id matches
    - returns a DataFrame with ANN_FIELDS (missing keys filled with "")
    """
    rows = []
    for aid in annotation_ids:
        try:
            js = sq.get(f"/api/annotation/{int(aid)}").execute().json()
            if not isinstance(js, dict) or js.get("id") is None:
                print(f"[warn] annotation id {aid} not found")
                continue

            # media check
            media_from_ann = get_by_dotted(js, "point.media.id", default=get_by_dotted(js, "point.media_id", default=None))
            if media_from_ann is None or int(media_from_ann) != int(media_id):
                print(f"[warn] annotation {aid} belongs to media {media_from_ann}, not {media_id}; skipping")
                continue

            # annset check (optional)
            if annset_id is not None and ENFORCE_ANNSET:
                annset_from_ann = get_by_dotted(js, "annotation_set_id", default=get_by_dotted(js, "point.annotation_set_id", default=None))
                if annset_from_ann is None or int(annset_from_ann) != int(annset_id):
                    print(f"[warn] annotation {aid} belongs to set {annset_from_ann}, not {annset_id}; skipping")
                    continue

            # flatten to requested schema
            row = {k: get_by_dotted(js, k, "") for k in ann_fields}
            rows.append(row)
        except Exception as e:
            print(f"[warn] GET /api/annotation/{aid} failed: {e}")

    if not rows:
        return pd.DataFrame(columns=ann_fields)
    df = pd.DataFrame(rows, columns=ann_fields)
    return df

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    sq = get_sqapi()

    # 1) Media metadata -> media_<MEDIA_ID>.csv
    media = sq.get(f"/api/media/{int(MEDIA_ID)}").execute().json()
    media_row = {k: get_by_dotted(media, k, "") for k in MEDIA_FIELDS}
    media_csv = OUT_DIR / f"media_{MEDIA_ID}.csv"
    pd.DataFrame([media_row]).to_csv(media_csv, index=False)
    print(f"[media] wrote {media_csv}")

    # 2) Download image (optional)
    local_img_path = ""
    if DOWNLOAD_IMAGE:
        url = media.get("path_best")
        if not url:
            raise RuntimeError("No path_best for this media; cannot download")
        img = SQMediaObject(url, media_type="image", media_id=int(MEDIA_ID)).data()
        name = f"{MEDIA_ID}_{os.path.basename(urlparse(url).path)}"
        local_img_path = str(OUT_DIR / name)
        import cv2
        cv2.imwrite(local_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[img  ] saved {local_img_path}")

    # 3) Annotations
    if ANNOTATION_IDS:  # robust per-ID GET path
        ann_df = fetch_annotations_by_ids_for_media_via_get(
            sq, int(MEDIA_ID), [int(x) for x in ANNOTATION_IDS],
            ANNSET_ID, ANN_FIELDS
        )
        subset_tag = "__subset"
    else:  # fetch all annotations on this media (optionally within ANNSET_ID)
        ann_df = export_all_annotations_for_media(
            sq, int(MEDIA_ID), ANNSET_ID, ANN_FIELDS
        )
        subset_tag = ""

    if ann_df.empty:
        print("[ann  ] no annotations returned for this image with current filters.")
    else:
        # provenance
        ann_df["requested_media_id"] = int(MEDIA_ID)
        if ANNSET_ID is not None and "annotation_set_id" not in ann_df.columns:
            ann_df["annotation_set_id"] = int(ANNSET_ID)
        if local_img_path:
            ann_df["local_image_path"] = local_img_path

        out_csv = OUT_DIR / f"annotations_for_media_{MEDIA_ID}{subset_tag}.csv"
        ann_df.to_csv(out_csv, index=False)
        print(f"[ann  ] wrote {out_csv} (rows={len(ann_df)})")

    # 4) Optional overlay with markers
    if DRAW_OVERLAY and DOWNLOAD_IMAGE and local_img_path and not ann_df.empty:
        try:
            import cv2
            img_bgr = cv2.imread(local_img_path, cv2.IMREAD_COLOR)
            count = 0
            for _, r in ann_df.iterrows():
                try:
                    x, y = float(r.get("point.x", 0)), float(r.get("point.y", 0))
                    cv2.drawMarker(img_bgr, (int(round(x)), int(round(y))),
                                   (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                                   markerSize=15, thickness=2)
                    count += 1
                except Exception:
                    continue
            overlay_path = str(OUT_DIR / f"{Path(local_img_path).stem}_overlay.jpg")
            cv2.imwrite(overlay_path, img_bgr)
            print(f"[viz ] saved overlay with {count} markers → {overlay_path}")
        except Exception as e:
            print(f"[viz ] skipped overlay (reason: {e})")
