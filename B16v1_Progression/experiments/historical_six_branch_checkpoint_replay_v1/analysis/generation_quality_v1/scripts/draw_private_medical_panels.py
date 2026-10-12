"""Restricted fixed-node panels; no downloads or external visual review; REAL pixels withheld."""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from b16_quality_common import setup, read_json, write_csv, atomic_json, now
from check_decode_and_image_stats import native_rgb


def main(output_root):
    root=setup(output_root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    dest=root/'figures/private_medical_panels';os.chmod(dest,0o700)
    q=pd.read_parquet(root/'manifests/COMBINED_QUALITY_EVAL_FROZEN_QUEUE_V1.parquet')
    nodes=pd.read_csv(root/'manifests/PRECOMMITTED_VISUAL_PANEL_NODES_V1.csv');rows=[]
    for r in nodes.itertuples():
        selected=q[q.anonymous_node_key==r.anonymous_node_key]
        imgs={b:[native_rgb(p)[0] for p in selected[selected.branch==b].sort_values('image_index').png_abs_path]
              for b in ['OP_CPFM_ON','OP_CPFM_OFF','FULL_B16V1_HIST','PLAIN_SD_HIST']}
        assert all(len(v)==6 for v in imgs.values())
        group='MedicalPlate01' if r.display_section=='MAIN_FIXED_FIRST6' else 'MedicalPlate02'
        specs=[(group,['OP_CPFM_ON','OP_CPFM_OFF']),('MedicalPlate04_AUX',['FULL_B16V1_HIST','PLAIN_SD_HIST'])]
        for group,branches in specs:
            fig,axes=plt.subplots(2,6,figsize=(12,4.5),layout='constrained')
            for j,b in enumerate(branches):
                for i in range(6):
                    ax=axes[j,i];ax.imshow(imgs[b][i],vmin=0,vmax=255,aspect='equal');ax.axis('off')
                    ax.set_title(('ON' if b=='OP_CPFM_ON' else 'OFF' if b=='OP_CPFM_OFF' else 'Full aux.' if b=='FULL_B16V1_HIST' else 'Plain aux.')+f' / index {i}',fontsize=9)
            fig.suptitle(r.panel_node_code+' | All six frozen seeds | No target ground truth\nReal historical pixels withheld: de-identification not yet verified; server-only restricted comparison',fontsize=10)
            path=dest/f'{group}_{r.panel_node_code}.png';fig.savefig(path,dpi=160);plt.close(fig);os.chmod(path,0o600)
            with Image.open(path) as im:shape=im.size
            rows.append(dict(group=group,panel_node_code=r.panel_node_code,file=str(path.relative_to(root)),displayed_generated_images=12,
                             real_context_status='WITHHELD_PENDING_VERIFIED_DEIDENTIFICATION',source_pixel_change='none; display-scale only',
                             image_dimensions=str(shape),technical_QA='AUTO_RENDER_ONLY',agent_visual_review='NOT_RUN',human_expert_review='NOT_RUN',public_or_transferred=False))
        fig,axes=plt.subplots(3,6,figsize=(12,6.7),layout='constrained')
        for i in range(6):
            a,b=imgs['OP_CPFM_ON'][i],imgs['OP_CPFM_OFF'][i];diff=np.abs(a.astype(float)-b).mean(2)
            axes[0,i].imshow(diff,cmap='magma',vmin=0,vmax=255,aspect='equal');axes[0,i].set_title(f'RGB MAE map / index {i}',fontsize=9)
            axes[1,i].imshow(a[192:320,192:320],vmin=0,vmax=255,aspect='equal');axes[1,i].set_title('ON fixed center crop',fontsize=9)
            axes[2,i].imshow(b[192:320,192:320],vmin=0,vmax=255,aspect='equal');axes[2,i].set_title('OFF fixed center crop',fontsize=9)
            for ax in axes[:,i]:ax.axis('off')
        fig.suptitle(r.panel_node_code+' | Difference map shared0-255 scale | Center crop (192:320,192:320), not anatomy ROI\nNo enhancement, no physical-mm scale, no medical authenticity interpretation',fontsize=10)
        path=dest/f'MedicalPlate03_DIFFERENCE_{r.panel_node_code}.png';fig.savefig(path,dpi=160);plt.close(fig);os.chmod(path,0o600)
        rows.append(dict(group='MedicalPlate03_DIFFERENCE',panel_node_code=r.panel_node_code,file=str(path.relative_to(root)),displayed_generated_images=12,
                         real_context_status='NOT_APPLICABLE',source_pixel_change='absolute RGB difference + fixed center crops; originals unchanged',
                         image_dimensions=str(Image.open(path).size),technical_QA='AUTO_RENDER_ONLY',agent_visual_review='NOT_RUN',human_expert_review='NOT_RUN',public_or_transferred=False))
    write_csv(dest/'PRIVATE_MEDICAL_PANEL_MANIFEST_V1.csv',rows);os.chmod(dest/'PRIVATE_MEDICAL_PANEL_MANIFEST_V1.csv',0o600)
    atomic_json(root/'figures/export_qa/PRIVATE_MEDICAL_PANEL_TECHNICAL_QA_V1.json',dict(status='RESTRICTED_PANELS_RENDERED_WITH_REAL_CONTEXT_WITHHELD',
                groups=4,node_panels=len(rows),fixed_nodes=len(nodes),all_six_seeds_preserved=True,generated_pixel_preprocess='no brightness/contrast change; display only',
                real_pixels_rendered=0,reason='No verified OCR/de-identification asset available; no unverified REAL pixels displayed',
                medical_visual_review='NOT_RUN',clinical_expert_review='NOT_RUN',medical_images_transferred=0,new_generator_samples=0,updated_utc=now()))
    (dest/'INTERNAL_REVIEW_INSTRUCTIONS_V1.md').write_text('''# Restricted server-only review

These are fixed saved-image comparison panels, not newly sampled images. Use an authorized local server viewer to open MedicalPlate01/02/03/04 files; no image was downloaded or transmitted in this task. Do not export or upload these panels. REAL pixels are deliberately withheld until display-specific de-identification is verified; their native originals and statistics are unchanged. The original frozen REAL reference manifest remains private on the server for an appropriately authorized reviewer. Image names, titles and blinded templates contain no actual patient IDs or dates. Generated panels have not undergone expert or cross-environment medical visual review and are not approved for public release.
''',encoding='utf-8')
    print({'private_node_panels':len(rows),'fixed_nodes':len(nodes),'medical_transfer':0,'real_pixels_withheld':True},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);a=p.parse_args();main(a.output_root)
