"""
评估指标计算模块
"""

class COCOEvalMetrics:
    """用于计算图像描述生成任务的评估指标"""
    
    @staticmethod
    def calculate_bleu(references, hypotheses):
        """
        计算BLEU分数
        
        Args:
            references: 参考描述列表的字典，格式为 {image_id: [caption1, caption2, ...]}
            hypotheses: 生成描述的字典，格式为 {image_id: [caption]}
            
        Returns:
            tuple: (BLEU-1, BLEU-2, BLEU-3, BLEU-4) 分数
        """
        try:
            from pycocoevalcap.bleu.bleu import Bleu
        except ImportError:
            raise ImportError("请安装评估工具: pip install pycocoevalcap")
        
        scorer = Bleu(n=4)
        scores, _ = scorer.compute_score(references, hypotheses)
        return scores
    
    @staticmethod
    def calculate_cider(references, hypotheses):
        """
        计算CIDEr分数
        
        Args:
            references: 参考描述列表的字典，格式为 {image_id: [caption1, caption2, ...]}
            hypotheses: 生成描述的字典，格式为 {image_id: [caption]}
            
        Returns:
            float: CIDEr分数
        """
        try:
            from pycocoevalcap.cider.cider import Cider
        except ImportError:
            raise ImportError("请安装评估工具: pip install pycocoevalcap")
        
        scorer = Cider()
        score, _ = scorer.compute_score(references, hypotheses)
        return score
    
    @staticmethod
    def evaluate_captions(references, hypotheses):
        """
        评估生成的图像描述
        
        Args:
            references: 参考描述列表的字典，格式为 {image_id: [caption1, caption2, ...]}
            hypotheses: 生成描述的字典，格式为 {image_id: [caption]}
            
        Returns:
            dict: 包含各项指标的字典
        """
        bleu_scores = COCOEvalMetrics.calculate_bleu(references, hypotheses)
        cider_score = COCOEvalMetrics.calculate_cider(references, hypotheses)
        
        return {
            'bleu1': bleu_scores[0],
            'bleu2': bleu_scores[1],
            'bleu3': bleu_scores[2],
            'bleu4': bleu_scores[3],
            'cider': cider_score
        } 