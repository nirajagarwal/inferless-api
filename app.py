import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

MODEL_WEIGHTS_DIR =  "/var/nfs-mount/sam-volume"
MODEL_WEIGHTS_FILE = "sam_vit_b_01ec64.pth"

class InferlessPythonModel:
    def initialize(self):
        #model_path = './data/sam_vit_b_01ec64.pth'
        model_path = MODEL_WEIGHTS_DIR
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry['vit_b'](checkpoint=model_path)
        _ = sam.to(device=device)
        self.generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask", min_mask_region_area=50)

    def infer(self,inputs):
        image_rgb = inputs["image_rgb"]
        masks = self.generator.generate(image_rgb)
        return {"masks": masks}

    def finalize(self,args):
          pass
