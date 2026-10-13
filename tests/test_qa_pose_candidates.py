import numpy as np
from types import SimpleNamespace
from sonkkeut_vision.plane import ScreenPlaneEstimator

COMPLETE=np.float32([[100,80],[540,80],[540,560],[100,560]])

class Tensor:
    def __init__(self, data): self.data=np.asarray(data)
    def cpu(self): return self
    def numpy(self): return self.data

def estimator(points, scores, confidence):
    result=SimpleNamespace(boxes=SimpleNamespace(conf=Tensor(scores), __len__=lambda self:len(scores)),
                           keypoints=SimpleNamespace(xy=Tensor(points),conf=Tensor(confidence)))
    class Boxes:
        conf=Tensor(scores)
        def __len__(self): return len(scores)
    result.boxes=Boxes()
    class Model:
        def predict(self,*args,**kwargs): return [result]
    return ScreenPlaneEstimator(Model(),refiner=None)

def test_clipped_top_candidate_does_not_hide_complete_screen():
    clipped=COMPLETE.copy(); clipped[:,0]-=200
    actual=estimator([clipped,COMPLETE],[.99,.8],[[.9]*4,[.9]*4]).detect(np.zeros((640,640,3),np.uint8))
    np.testing.assert_allclose(actual[0],COMPLETE)

def test_low_keypoint_candidate_does_not_hide_complete_screen():
    actual=estimator([COMPLETE,COMPLETE],[.99,.8],[[.2]*4,[.9]*4]).detect(np.zeros((640,640,3),np.uint8))
    assert (actual[2]>.8).all()

def test_invalid_candidates_still_produce_framing_hint():
    clipped=COMPLETE.copy(); clipped[:,0]-=200
    actual=estimator([clipped],[.99],[[.9]*4]).detect(np.zeros((640,640,3),np.uint8))
    assert ScreenPlaneEstimator.missing_hint(actual[0],actual[2],(640,640,3)) is not None

def test_nonfinite_scores_and_points_are_rejected():
    assert estimator([COMPLETE],[np.nan],[[.9]*4]).detect(np.zeros((640,640,3),np.uint8)) is None
    invalid=COMPLETE.copy(); invalid[0,0]=np.nan
    assert estimator([invalid],[.9],[[.9]*4]).detect(np.zeros((640,640,3),np.uint8)) is None
