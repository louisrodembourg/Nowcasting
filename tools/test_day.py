#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clustering import hdbscan_daily as hd
from src.clustering import visualize_clusters as vc
from datetime import date

p = Path('data/parquet/la/la_2017_03_15.parquet')
print('PARQUET', p.exists(), p)
if not p.exists():
    raise SystemExit('Parquet missing')
config = vc._load_config('la')
print('CONFIG waiting polygons:', bool(getattr(config, 'waiting_allowed_polygons', None)))
cluster_df, prepared = hd.cluster_day(p, config=config)
print('CLUSTER_DF_NONE', cluster_df is None)
if cluster_df is not None:
    uniq_labels = cluster_df.select(['cluster_label']).unique().sort('cluster_label')
    print('UNIQ_LABELS_COUNT', len(uniq_labels))
    print(cluster_df.select(['MMSI','cluster_label','cluster_type']).head(5))
    out = Path('outputs/figures') / 'la_test_2017_03_15.html'
    vc.visualize_day(date(2017,3,15), 'la', config, out)
    print('SAVED', out.exists(), out)
else:
    print('NO CLUSTERS')
