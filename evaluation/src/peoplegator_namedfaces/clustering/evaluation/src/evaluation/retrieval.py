from peoplegator_namedfaces.clustering.evaluation.src.schemas import (
    PeopleGatorNamedFaces__GroundTruth,
    PeopleGatorNamedFaces__FaceRetrievalPrediction,
)

def create_eval_arrays(
    ground_truths: list[PeopleGatorNamedFaces__GroundTruth],
    predictions: list[PeopleGatorNamedFaces__FaceRetrievalPrediction],
):
    person_name_to_index = {gt: i for i, gt in enumerate(
        sorted(set(gt.person_name for gt in ground_truths)))
    }
    face_to_person_name = {gt.face: gt.person_name for gt in ground_truths}
    
    