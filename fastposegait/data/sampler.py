import math
import random
import numpy
import torch
import torch.distributed as dist
import torch.utils.data as tordata


class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            sample_indices = []
            pid_list = sync_random_sample_list(
                self.dataset.label_set, self.batch_size[0])

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                indices = sync_random_sample_list(
                    indices, k=self.batch_size[1])
                sample_indices += indices

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)


class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):

        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if batch_size % world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({})".format(
                world_size, batch_size))

        if batch_size != 1:
            complement_size = math.ceil(self.size / batch_size) * \
                batch_size
            indices += indices[:(complement_size - self.size)]
            self.size = complement_size

        batch_size_per_rank = int(self.batch_size / world_size)
        indx_batch_per_rank = []

        for i in range(int(self.size / batch_size_per_rank)):
            indx_batch_per_rank.append(
                indices[i*batch_size_per_rank:(i+1)*batch_size_per_rank])

        self.idx_batch_this_rank = indx_batch_per_rank[rank::world_size]

    def __iter__(self):
        yield from self.idx_batch_this_rank

    def __len__(self):
        return len(self.dataset)
    

class CommonSampler(tordata.sampler.Sampler):
    def __init__(self,dataset,batch_size,batch_shuffle):

        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        if isinstance(self.batch_size,int)==False:
            raise ValueError(
                "batch_size shoude be (B) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle
        
        self.world_size = dist.get_world_size()
        if self.batch_size % self.world_size !=0:
            raise ValueError("World size ({}) is not divisble by batch_size ({})".format(
                self.world_size, batch_size))
        self.rank = dist.get_rank() 
    
    def __iter__(self):
        while True:
            indices_list = list(range(self.size))
            sample_indices = sync_random_sample_list(
                    indices_list, self.batch_size, common_choice=True)
            total_batch_size =  self.batch_size
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]
            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)


class RandomTripletSampler(tordata.sampler.Sampler):
    '''
    This sampler is a trade-off of Random Sampler and Triplet Sampler. 
    Note that this sampler is not compatible with triplet loss with concise input formulation of [a, p, n].
    '''
    def __init__(self, dataset, batch_size, P_max=None, K_max=None, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle
        self.size = len(dataset)
        self.world_size = dist.get_world_size()
        
        if self.batch_size % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({})".format(
                self.world_size, self.batch_size))
        self.rank = dist.get_rank()
        # get all possible combination of PK
        if not P_max:
            P_max = batch_size 
        if not K_max:
            K_max = batch_size 
        self.comb_list = []
        for i in range(self.batch_size):
            p = i + 1
            k = int(self.batch_size / p)
            rest = int(self.batch_size % p)
            if p > 1 and k > 1 and p < P_max+1 and k < K_max+1:
                self.comb_list.append((p, k, rest))

    def __iter__(self):
        while True:
            # Random select PK
            idx = int(numpy.random.choice(range(len(self.comb_list)),size=1)) # select one combination idx
            # broadcast to other gpus
            idx = torch.tensor(idx)
            if torch.cuda.is_available():
                idx = idx.cuda()
            torch.distributed.broadcast(idx, src=0)
            # select one by idx
            batch_size =self.comb_list[idx]
            
            # select P
            sample_indices = []
            pid_list = sync_random_sample_list(
                self.dataset.label_set, batch_size[0])
            # select K
            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                indices = sync_random_sample_list(
                    indices, k=batch_size[1])
                sample_indices += indices
            # select rest
            all_indices_list = list(range(self.size))
            # remove selected idx
            for i in sample_indices:
                # add condition
                if i in all_indices_list:
                    all_indices_list.remove(i)
            # select rest from rest data
            sample_indices += sync_random_sample_list(
                    all_indices_list, batch_size[2], common_choice=True)
            
            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = batch_size[0] * batch_size[1]+batch_size[2]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices


def sync_random_sample_list(obj_list, k,common_choice=False):
    if common_choice:
        idx = numpy.random.choice(range(len(obj_list)),size=k,replace=True) 
        idx = torch.tensor(idx)
    if len(obj_list) < k:
        idx = random.choices(range(len(obj_list)), k=k) #Default replace=True 
        idx = torch.tensor(idx)
    else:
        idx = torch.randperm(len(obj_list))[:k]
    if torch.cuda.is_available():
        idx = idx.cuda()
    torch.distributed.broadcast(idx, src=0)
    idx = idx.tolist()
    return [obj_list[i] for i in idx]
