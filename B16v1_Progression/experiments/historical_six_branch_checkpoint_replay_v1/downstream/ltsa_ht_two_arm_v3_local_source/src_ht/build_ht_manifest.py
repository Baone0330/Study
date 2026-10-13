"""Native pre-target-year patient windows, no LOCF duplication and no target-year pixel reads."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from common import setup,config,jread,jout,cout,pout,metadata,postflight,state,now


def parse_exam_date(text):
    for fmt in ['%Y-%m-%d','%Y.%m.%d','%Y_%m_%d','%Y%m%d']:
        try:return datetime.strptime(text,fmt).date()
        except ValueError:pass
    return None


def relative_months(dates):
    assert dates and all(dates[i]<dates[i+1] for i in range(len(dates)-1))
    return [int((d-dates[0]).days/30.4375) for d in dates]


def allowed_history(date,target_year):return date.year<int(target_year)


def main(config_path):
    c=config(config_path);root=setup(c['project_private']);old=Path(c['source_old_ablation_private']);data=Path(c['raw_data_private']).resolve()
    done=root/'releases/REAL_MANIFEST_COMPLETE_V3.json'
    if done.exists():postflight(root);print('Frozen native manifest retained; no target-year reads');return
    rows=[];views=[];exams=[];missing=[];counts=[];index=0
    for split,fn,nexpect,expected_labels in [('train','ORIGINAL_TRAIN_SEQUENCE_V1.csv',860,{0:617,1:243}),('val','ORIGINAL_VAL_SEQUENCE_V1.csv',189,{0:131,1:58})]:
        d=pd.read_csv(old/'manifests'/fn,dtype={'patient_id':str});assert d.split.eq(split).all()
        group=d.groupby('patient_id',sort=True);assert len(group)==nexpect
        assert group.target_label.first().value_counts().to_dict()==expected_labels
        for pid,g in group:
            assert g.target_label.nunique()==g.target_year.nunique()==1;y=int(g.target_label.iloc[0]);year=int(g.target_year.iloc[0]);assert y in [0,1]
            folders=[data/str(lab)/pid for lab in [0,1] if (data/str(lab)/pid).is_dir()];assert len(folders)==1 and folders[0].parent.name==str(y)
            folder=folders[0];dated=[];invalid=[]
            # Only list date directory names; do not enumerate any target-year image/report files.
            for p in sorted(folder.iterdir()):
                if not p.is_dir():continue
                date=parse_exam_date(p.name)
                if date is None:invalid.append(p.name)
                else:dated.append((date,p))
            assert len({d for d,p in dated})==len(dated),'Duplicate real exam dates need explicit provenance resolution'
            target_dates=[d for d,p in dated if d.year==year];history=[(d,p) for d,p in dated if allowed_history(d,year)]
            if invalid:missing.append(dict(patient_key_private=pid,split=split,reason='UNPARSED_REAL_EXAM_DATE',folder_names_private=invalid))
            if not target_dates:missing.append(dict(patient_key_private=pid,split=split,reason='SOURCE_TARGET_YEAR_NOT_IN_RAW_DATE_DIRECTORY_METADATA'))
            patient_exams=[];patient_view_paths=[]
            for date,p in sorted(history):
                native=sorted(z for z in p.rglob('*') if z.is_file() and z.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.tif','.tiff'])
                if not native:
                    missing.append(dict(patient_key_private=pid,split=split,reason='EMPTY_HISTORICAL_EXAM',source_path_private=str(p)));continue
                e=len(patient_exams);patient_exams.append(dict(date=date.isoformat(),paths=[str(z) for z in native],source_type='REAL',views=len(native)))
                exams.append(dict(patient_index=index,patient_key_private=pid,split=split,exam_index=e,exam_date_private=date.isoformat(),native_views=len(native),source_type='REAL'))
                for z in native:views.append(dict(patient_index=index,patient_key_private=pid,split=split,exam_index=e,exam_date_private=date.isoformat(),native_path_private=str(z),source_type='REAL',target_year=year))
                patient_view_paths.extend(native)
            legal=bool(patient_exams) and not invalid and bool(target_dates)
            if not patient_exams:missing.append(dict(patient_key_private=pid,split=split,reason='NO_LEGAL_PRE_TARGET_HISTORY'))
            dates=[datetime.fromisoformat(e['date']).date() for e in patient_exams];months=relative_months(dates) if dates else []
            rec=dict(patient_index=index,patient_key_private=pid,split=split,target_year=year,target_HT=y,observed_history_exam_count=len(patient_exams),observed_history_view_count=len(patient_view_paths),
                     first_history_exam_date=dates[0].isoformat() if dates else None,last_history_exam_date=dates[-1].isoformat() if dates else None,
                     latest_real_target_exam_date_metadata_only=max(target_dates).isoformat() if target_dates else None,canonical_reference_branch='FROZEN_HISTORICAL_SHARED_TRAIN_VAL',
                     eligibility_status='ELIGIBLE' if legal else 'ENGINEERING_UNRESOLVED',exclusion_reason='' if legal else 'see private input difference ledger',
                     exams_private_json=json.dumps(patient_exams),relative_months_private_json=json.dumps(months),native_cache_private=str(root/'cache/real224'/f'window_{index:04d}.npy'))
            rows.append(rec);index+=1
    master=pd.DataFrame(rows);ev=pd.DataFrame(exams);iv=pd.DataFrame(views)
    assert len(master)==1049 and master.patient_key_private.nunique()==1049
    assert set(master[master.split=='train'].patient_key_private).isdisjoint(master[master.split=='val'].patient_key_private)
    pout(root/'manifests_private/HT_LTSA_CANONICAL_PATIENT_WINDOWS_V1.parquet',master)
    pout(root/'manifests_private/REAL_NATIVE_EXAMS_V3.parquet',ev);pout(root/'manifests_private/REAL_NATIVE_VIEWS_V3.parquet',iv)
    jout(root/'audit/PRIVATE_INPUT_DIFFERENCE_LEDGER_V3.json',missing)
    for split,g in master.groupby('split'):
        counts.append(dict(split=split,patients=len(g),HT0=int(g.target_HT.eq(0).sum()),HT1=int(g.target_HT.eq(1).sum()),real_exams=int(g.observed_history_exam_count.sum()),native_views=int(g.observed_history_view_count.sum()),
                    max_exams=int(g.observed_history_exam_count.max()),max_views_per_patient=int(g.observed_history_view_count.max()),max_views_per_exam=int(ev[ev.split==split].native_views.max()),
                    max_relative_month=int(max([max(json.loads(x)) for x in g.relative_months_private_json if json.loads(x)],default=0)),
                    unresolvable_patients=int(g.eligibility_status.ne('ELIGIBLE').sum()),valid_time_tokens=int(g.observed_history_exam_count.sum()),synthetic_views=0))
    cout(root/'tables/REAL_COHORT_NATIVE_SUPPORT_V3.csv',counts)
    if missing or master.eligibility_status.ne('ELIGIBLE').any():
        state(root,'CANONICAL_INPUT_ENGINEERING_UNRESOLVED',patient_windows=len(master),issues=len(missing));print({'cohort_counts':counts,'input_issue_records':len(missing),'training_started':False});raise RuntimeError('Native historical identity/date/support issue; no silent exclusions or new split. Inspect private ledger')
    # Freeze protection of any additional native views (outside historical fixedK slot support).
    protection=jread(root/'audit/PROTECTED_SOURCE_METADATA_V3.json');have={r['path'] for r in protection}
    for p in iv.native_path_private:
        if p not in have:protection.append(metadata(p));have.add(p)
    jout(root/'audit/PROTECTED_SOURCE_METADATA_V3.json',protection)
    for i,r in enumerate(master.itertuples(index=False)):
        dest=Path(r.native_cache_private);paths=[Path(p) for e in json.loads(r.exams_private_json) for p in e['paths']];arrays=[]
        for path in paths:
            with Image.open(path) as im:
                im.load();arrays.append(np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)).copy())
        a=np.stack(arrays);assert a.shape==(r.observed_history_view_count,224,224,3) and a.dtype==np.uint8
        temp=dest.with_suffix('.tmp.npy');np.save(temp,a);temp.replace(dest)
        if (i+1)%100==0:print(f'REAL_NATIVE_CACHE {i+1}/{len(master)} patient windows',flush=True)
    jout(done,dict(status='COMPLETE',counts=counts,patient_windows=len(master),target_year_pixel_reads=0,official_Test_pixel_or_probability_reads=0,
                   canonical_source='separate historicalTrain andVal frozen CSVs',LOCF_not_new_exam=True,all_native_views=True,raw_content_fingerprint_used=False,completed_utc=now()))
    (root/'audit/CANONICAL_COHORT_IDENTITY_AUDIT_V3.md').write_text('# Canonical identity\n\nHistorical Train860(617/243),Val189(131/58) exact master identity and labels retained. Native dates/whole real exam sets reconstructed without LOCF copies. No Test manifest loaded; Train/Val disjoint; Official Test isolation inherited from protected source split audit, not a new Test-content audit.\n',encoding='utf-8')
    (root/'audit/REAL_INPUT_BOUNDARY_AUDIT_V3.md').write_text('# Input boundary\n\nOnly pre-target-year directories were enumerated/read for pixels. Target-year folder names provide exclusion metadata only; their contents are not opened. New private uint8RGB224 caches contain exactly selected native history views. Dataset returns numeric sample index, normalized images, valid exam grouping, observed relative months and supervised label; no target year/date/path/report values enter the predictor.\n',encoding='utf-8')
    postflight(root);state(root,'NATIVE_PATIENT_WINDOWS_AND_INPUT_CACHES_FROZEN',patient_windows=len(master),counts=counts);print({'native_counts':counts,'target_year_pixel_reads':0,'Test_access':0},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)
