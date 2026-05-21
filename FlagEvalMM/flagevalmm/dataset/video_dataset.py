import os.path as osp
from typing import Dict, Any
from flagevalmm.registry import DATASETS
from flagevalmm.dataset.vqa_base_dataset import VqaBaseDataset


@DATASETS.register_module()
class VideoDataset(VqaBaseDataset):
    def __getitem__(self, index: int) -> Dict[str, Any]:
        annotation = self.annotations[index]
        video_path = []
        if isinstance(annotation["video_path"], list):
            for path in annotation["video_path"]:
                # 如果是绝对路径，直接使用，否则拼接data_root
                if osp.isabs(path):
                    video_path.append(path)
                else:
                    video_path.append(osp.join(self.data_root, path))
        else:
            path = annotation["video_path"]
            if osp.isabs(path):
                video_path.append(path)
            else:
                video_path.append(osp.join(self.data_root, path))
                
        ret = {
            "video_path": video_path,
            "question": self.build_prompt(annotation, []),
            "question_id": str(annotation["question_id"]),
            "type": annotation["question_type"],
        }
        if self.with_label and "answer" in annotation:
            ret["label"] = annotation["answer"]
        return ret

    def meta_info(self) -> Dict[str, Any]:
        return {"name": self.name, "length": len(self.annotations), "type": "video_qa"}
