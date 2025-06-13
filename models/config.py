"""
模型配置模块
"""

class ModelConfig:
    """模型配置类，用于存储模型参数"""
    
    def __init__(
        self,
        vision_model_name="openai/clip-vit-base-patch16",
        language_model_name="Qwen/Qwen3-0.6B",
        vision_select_layer=-1,
        projector_type="mlp",
        mm_use_im_start_end=False,
        image_token="<image>",
        image_aspect_ratio="square",
    ):
        # 模型名称
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        
        # 视觉模型配置
        self.vision_select_layer = vision_select_layer
        
        # 连接器配置
        self.projector_type = projector_type
        
        # 多模态标记配置
        self.mm_use_im_start_end = mm_use_im_start_end
        self.image_token = image_token
        self.image_aspect_ratio = image_aspect_ratio
        
        # 隐藏层大小将在模型初始化时设置
        self.vision_hidden_size = None
        self.text_hidden_size = None
    
    def update_sizes(self, vision_hidden_size, text_hidden_size):
        """更新隐藏层大小"""
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        return self
        
    def to_dict(self):
        """将配置转换为字典"""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置"""
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config 