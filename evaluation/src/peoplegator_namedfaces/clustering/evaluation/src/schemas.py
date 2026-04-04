from itertools import pairwise
from typing import Literal
from pydantic import BaseModel
from scipy.constants import micro

InteragreementStrategy = Literal["union",
                                 "intersection", "majority", "probabilistic"]

class PeopleGatorNamedFaces__UniqueFace(BaseModel):
    face: str

class PeopleGatorNamedFaces__GroundTruth(BaseModel):
    person_name: str
    annotator: str
    library: str
    document: str
    page: str
    crop_name: str
    face: str
    page_height: int
    page_width: int
    page_left: int
    page_top: int
    height: int
    width: int
    page_keypoints: list[list[float]]

class PeopleGatorNamedFaces__ClusterPrediction(BaseModel):
    face: str
    cluster: int
    cluster_score: float | None = None

class PeopleGatorNamedFaces__PairwiseSparseScorePrediction(BaseModel):
    face_1: str
    face_2: str
    score: float

class PeopleGatorNamedFaces__PairwiseDenseScorePrediction(BaseModel):
    faces: list[str]
    scores: list[list[float]]

class PeopleGatorNamedFaces__FaceRetrievalPrediction(BaseModel):
    query_face: str
    retrieved_faces: list[str]
    retrieved_faces_scores: list[float] | None = None

class PeopleGatorNamedFaces__NameRetrievalPrediction(BaseModel):
    query_name: str
    retrieved_faces: list[str]
    retrieved_faces_scores: list[float] | None = None


class PeopleGatorNamedFaces__Point2D(BaseModel):
    x: float
    y: float
    page_width: int
    page_height: int

class PeopleGatorNamedFaces__FacePointerPrediction(BaseModel):
    person_name: str
    library: str
    document: str
    page: str
    points: list[PeopleGatorNamedFaces__Point2D]


class PeopleGatorNamedFaces__PairwiseReport(BaseModel):    
    true_positives: float
    false_positives: float
    true_negatives: float
    false_negatives: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    jaccard: float
    fowlkes_mallows: float
    rand_index: float
    adjusted_rand_index: float

class PeopleGatorNamedFaces__AssignmentReport(BaseModel):
    num_samples: float
    true_positives: float
    false_positives: float
    true_negatives: float
    false_negatives: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float

class PeopleGatorNamedFaces__ClusterEvaluationReport(BaseModel):
    adjusted_mutual_info_score: float
    adjusted_rand_score: float
    calinski_harabasz_score: float
    completeness_score: float
    davies_bouldin_score: float
    dunn_index: float
    fowlkes_mallows_index: float
    homogeneity_score: float
    mutual_info_score: float
    normalized_mutual_info_score: float
    rand_score: float
    v_measure_score: float

__all__ = [
    "PeopleGatorNamedFaces__GroundTruth",
    "PeopleGatorNamedFaces__ClusterPrediction",
    "PeopleGatorNamedFaces__PairwiseSparseScorePrediction",
    "PeopleGatorNamedFaces__PairwiseDenseScorePrediction",
    "PeopleGatorNamedFaces__FaceRetrievalPrediction",
    "PeopleGatorNamedFaces__NameRetrievalPrediction",
    "PeopleGatorNamedFaces__ClusterEvaluationReport",
    "InteragreementStrategy"
]
