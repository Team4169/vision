
# Before you run the Python code snippet below, run the following command:
# pip install roboflow autodistill autodistill_grounding_dino

from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill.helpers import sync_with_roboflow

BOX_THRESHOLD = 0.5
CAPTION_ONTOLOGY = {
    "a red-orange ring": "note"
}
TEXT_THRESHOLD = 0.70

model = GroundingDINO(
    ontology=CaptionOntology(CAPTION_ONTOLOGY),
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

sync_with_roboflow(
    workspace_id="MKVa2Gd8aFTG6LeQnQv7NwvUErp2",
    workspace_url="frc-dataset-colab",
    project_id = "frc-2024-gk8et",
    batch_id = "OiqLvdyZQdYfcQbYsoUN",
    model = model
)

