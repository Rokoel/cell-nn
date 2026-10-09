from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from fish_viewer.pipeline.segmentation import Cancelled
from fish_viewer.proposals.recovery import (RecoveryConfig, local_surface_evidence, evidence_passes,
    crop_prediction, recover, candidate_labels)
from fish_viewer.refinement.config import RefinementConfig
from fish_viewer.refinement.inference import surface_evidence


def sphere_scene(amplitude=.08):
    spacing=np.array([.2]*3)
    grid=np.indices((48,)*3)
    radius=np.linalg.norm((grid-23.5)*spacing[:,None,None,None],axis=0)
    image=(amplitude*np.exp(-((radius-2.3)/.15)**2)).astype(np.float32)
    prepared=SimpleNamespace(image=image,spacing=tuple(spacing),origin=(.1,)*3)
    proposal=dict(proposal_id=1,score=.8,center_xyz=[4.8]*3,box_low_xyz=[2.3]*3,box_high_xyz=[7.3]*3)
    return prepared,proposal,radius<2.3


class MembraneModel:
    def __call__(self,inputs,training=False):
        rim=inputs[0,...,0]>.012
        mask=ndi.binary_fill_holes(rim)
        return np.where(mask[None,...,None],15.,-15.)


def rejection(proposal):
    return dict(proposal,cell_id=None,accepted=False,rejection_reason='Weak membrane contrast against local context')


def test_dim_membrane_passes_local_evidence_but_constant_noise_and_empty_regions_do_not():
    prepared,proposal,mask=sphere_scene(.06)
    visible=ndi.gaussian_filter(prepared.image,1.5)
    assert surface_evidence(mask,visible,prepared.spacing)['surface_contrast']<.04
    config=RecoveryConfig()
    assert evidence_passes(local_surface_evidence(mask,visible,prepared.spacing,config),config)
    rng=np.random.default_rng(42)
    for image in (np.zeros_like(visible),np.ones_like(visible)*.1,
                  ndi.gaussian_filter(rng.uniform(0,.15,visible.shape),1.5)):
        assert not evidence_passes(local_surface_evidence(mask,image,prepared.spacing,config),config)


def test_local_recovery_requires_visible_membrane_not_a_model_hallucination():
    prepared,proposal,mask=sphere_scene()
    config=replace(RefinementConfig(),patch_size=32)
    candidate=crop_prediction(prepared,proposal,MembraneModel(),config)
    labels=np.zeros(prepared.image.shape,np.uint32)
    labels,rows,summary=recover(prepared,[proposal],MembraneModel(),config,labels,[rejection(proposal)],
                               {1:candidate},retries=False)
    assert summary['local']==1 and rows[0]['accepted'] and labels.any()
    prepared.image[:]=0
    labels,rows,summary=recover(prepared,[proposal],MembraneModel(),config,np.zeros_like(labels),[rejection(proposal)],
                               {1:candidate},retries=False)
    assert not labels.any() and not rows[0]['accepted']


def test_expanded_crop_recovers_stable_cell_and_keeps_existing_labels():
    prepared,proposal,mask=sphere_scene(.2)
    proposal.update(box_low_xyz=[3.7]*3,box_high_xyz=[5.9]*3)
    config=replace(RefinementConfig(),patch_size=32,margin_um=0)
    original=crop_prediction(prepared,proposal,MembraneModel(),config)
    labels=np.zeros(prepared.image.shape,np.uint32);labels[:3,:3,:3]=1
    before=labels==1
    labels,rows,summary=recover(prepared,[proposal],MembraneModel(),config,labels,[rejection(proposal)],{1:original},
        config=replace(RecoveryConfig(),retry_margin_um=2.),local=False)
    assert summary['retry']==1 and rows[0]['recovery']['stable_pairs']>=2
    assert np.array_equal(labels==1,before)


def test_guidance_requires_high_confidence_signal_inside_final_supported_cell():
    prepared,proposal,mask=sphere_scene(.2)
    config=replace(RefinementConfig(),patch_size=32)
    candidate=crop_prediction(prepared,proposal,MembraneModel(),config)
    for xyz,score,expected in [([4.8]*3,.95,True),([4.8]*3,.8,False),([.3]*3,.99,False)]:
        signal=dict(cell_id=1,signal_id=1,channel=1,xyz=xyz,score=score)
        labels,rows,summary=recover(prepared,[proposal],MembraneModel(),config,
            np.zeros(prepared.image.shape,np.uint32),[rejection(proposal)],{1:candidate},signals=[signal],local=False)
        assert bool(labels.any())==expected
        assert summary['signal_guided']==int(expected)


def test_provisional_masks_exclude_accepted_cells_and_cancel_during_recovery():
    prepared,proposal,mask=sphere_scene()
    labels=mask.astype(np.uint32)
    region=(slice(0,48),)*3
    candidates={1:(region,mask,[23.5]*3)}
    assert not candidate_labels(labels,candidates,[rejection(proposal)]).any()
    class Cancel:
        def is_set(self):return True
    with pytest.raises(Cancelled):
        recover(prepared,[proposal],MembraneModel(),RefinementConfig(),labels,[rejection(proposal)],candidates,cancel=Cancel())


def test_signal_directed_seed_recovers_when_axis_only_retries_fail():
    prepared,proposal,mask=sphere_scene(.2)
    config=replace(RefinementConfig(),patch_size=32)
    class SeedSensitive(MembraneModel):
        def __call__(self,inputs,training=False):
            seed=np.unravel_index(np.argmax(inputs[0,...,1]),(32,)*3)
            if seed[1]<=15 or seed[2]<=15:return np.full((1,32,32,32,1),-15.)
            return super().__call__(inputs,training)
    model=SeedSensitive()
    labels,rows,summary=recover(prepared,[proposal],model,config,np.zeros(mask.shape,np.uint32),[rejection(proposal)],{},local=False)
    assert not labels.any()
    signal=dict(cell_id=1,signal_id=1,channel=1,xyz=[5.8,5.8,4.8],score=.95)
    labels,rows,summary=recover(prepared,[proposal],model,config,np.zeros(mask.shape,np.uint32),[rejection(proposal)],{},signals=[signal],local=False)
    assert labels.any() and summary['signal_guided']==1
    assert rows[0]['recovery']['stable_pairs']>=2


def test_recovery_configuration_rejects_invalid_values():
    with pytest.raises(ValueError):replace(RecoveryConfig(),signal_score=1.1).validate()
    with pytest.raises(ValueError):replace(RecoveryConfig(),seed_shift_um=-1).validate()


def test_partial_membrane_fragment_does_not_support_a_complete_cell():
    prepared,proposal,mask=sphere_scene(.2)
    prepared.image[:24]=0
    visible=ndi.gaussian_filter(prepared.image,1.5)
    assert not evidence_passes(local_surface_evidence(mask,visible,prepared.spacing,RecoveryConfig()),RecoveryConfig())


def test_raw_noise_floor_rejects_rim_with_insufficient_signal_to_noise():
    prepared,proposal,mask=sphere_scene(.06)
    rng=np.random.default_rng(123)
    raw=prepared.image+rng.uniform(0,.12,prepared.image.shape)
    visible=ndi.gaussian_filter(raw,1.5)
    evidence=local_surface_evidence(mask,visible,prepared.spacing,RecoveryConfig(),noise_image=raw)
    assert evidence['noise']>.02
    assert not evidence_passes(evidence,RecoveryConfig())


def test_signal_search_band_can_extend_beyond_provisional_contour_without_claiming_a_cell():
    prepared,proposal,mask=sphere_scene()
    region=(slice(0,48),)*3
    candidates={1:(region,mask,[23.5]*3)}
    provisional=candidate_labels(np.zeros_like(mask,np.uint32),candidates,[rejection(proposal)],prepared.spacing,.6)
    assert np.count_nonzero(provisional)>np.count_nonzero(mask)
    assert not np.any(provisional[~(ndi.distance_transform_edt(~mask,sampling=prepared.spacing)<=.6)])
