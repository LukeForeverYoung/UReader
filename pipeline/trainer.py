import torch.distributed as dist
import argparse
from functools import partial

import torch

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import Trainer

from pipeline.utils import batchify

# class MegatronPretrainingLengthBalancedSampler:
#     def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
#                  data_parallel_rank, data_parallel_size, drop_last=True, epoch_id=0):
#         # Keep a copy of input params for later use.
#         # 一定要保证使用的dataset本身是长度有序的!!!!!!
#         self.total_samples = 
        
#         self.consumed_samples = consumed_samples # 不使用
#         # 一个完整模型的batchsize大小
#         self.micro_batch_size = micro_batch_size
#         # 多少个数据并行模型
#         self.data_parallel_rank = data_parallel_rank
#         self.micro_batch_times_data_parallel_size = \
#             self.micro_batch_size * data_parallel_size
#         self.data_parallel_size = data_parallel_size
#         self.drop_last = drop_last

#         # Sanity checks.
#         assert self.total_samples > 0, \
#             'no sample to consume: {}'.format(self.total_samples)
#         # assert self.consumed_samples < self.total_samples, \
#         #     'no samples left to consume: {}, {}'.format(self.consumed_samples,
#         #                                                 self.total_samples)
#         from icecream import ic
#         ic(self.micro_batch_size)
#         ic(self.micro_batch_times_data_parallel_size)
#         assert self.micro_batch_size > 0
#         assert data_parallel_size > 0
#         assert self.data_parallel_rank < data_parallel_size, \
#             'data_parallel_rank should be smaller than data size: {}, ' \
#             '{}'.format(self.data_parallel_rank, data_parallel_size)

#         # 构造长度均衡的bucket
#         sorted_indices = list(range(total_samples))
#         bucket_len = self.micro_batch_times_data_parallel_size # global batch的长度
#         divide_point = len(sorted_indices) - len(sorted_indices) % bucket_len
#         sorted_indices = sorted_indices[:divide_point]
#         self.sorted_batch_indices = [sorted_indices[i: i + bucket_len] for i in
#                                 range(0, len(sorted_indices), bucket_len)]
#         self.epoch_id = epoch_id
      
    
#     def __len__(self):

#         return self.total_samples // self.data_parallel_size


#     def get_start_end_idx(self):
#         start_idx = self.data_parallel_rank * self.micro_batch_size
#         end_idx = start_idx + self.micro_batch_size
#         return start_idx, end_idx

#     def __iter__(self):
#         # 一种思路是 把数据有序排好 根据global batch size分成N份并shuffle 分好后重新组合成
#         # 这个seed要保证每个node一致
#         seed = [41, self.epoch_id]

#         def shuffle_with_seed(batches):
#             with numpy_seed(*seed):
#                 np.random.shuffle(batches)
#             return batches

#         # 将bucket打乱
#         batch_indices = shuffle_with_seed(list(range(len(self.sorted_batch_indices))))
        
#         # 计算起始bucket号码
#         #current_bucket_index = self.consumed_samples//self.micro_batch_times_data_parallel_size
#         # Megatron的sampler会调用consumed_samples 但我们的alicemind_gpt直接传入dataset 似乎忽略了这个参数 因此这里也同样忽略其作用 每次创建sampler从头开始迭代数据
#         current_bucket_index = 0
#         # Last batch will be dropped if drop_last is not set False
   
#         for idx in range(current_bucket_index, len(self.sorted_batch_indices)):
            
#             batch = self.sorted_batch_indices[batch_indices[idx]]
#             # 对batch内做shuffle 保证每个rank长度分布较为均匀
#             batch = shuffle_with_seed(batch) 
#             ## TODO 这里shuffle会有一点性能问题 因为之后还需要把数据进一步切分 如果当前batch比较大 那么长度跨度也会更大 使得切出来的每个sub batch的最大长度也很大. 解决方法是传入ga频率 把batch首先切分成(ga_size*world_size) * micro_batch_size，对ga_size做shuffle使得
#             ## 
#             # 获取当前dp下所有子卡的batch idx
#             start_idx, end_idx = self.get_start_end_idx()            
#             yield from batch[start_idx:end_idx]
#          # 处理剩余数据 这份代码默认会丢弃最后不能整除的数据 
#         # TODO 支持使用剩余数据
#         # Check the last partial batch and see drop_last is set
#         # if len(batch) > 0 and not self.drop_last:
#         #     start_idx, end_idx = self.get_start_end_idx()
#         #     yield batch[start_idx:end_idx]
    
class CustomTrainer(Trainer):
    
    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        sampler = DistributedSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            pin_memory=False,
            collate_fn=batchify)


    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        dataset = self.eval_dataset
        sampler = DistributedSampler(dataset, shuffle=False)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            pin_memory=False,
            collate_fn=batchify)