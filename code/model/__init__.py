import os
from importlib import import_module
from collections import OrderedDict
import mindspore as ms
import mindspore.nn as nn

class Model:
    def __init__(self, args, ckp):
        print('Making model...')
        self.args = args
        self.cpu = args.cpu
        self.device = 'cpu' if args.cpu else 'cuda'
        self.n_GPUs = args.n_GPUs
        self.save_middle_models = args.save_middle_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args)
        
        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu,
            args=args
        )
        print(self.get_model(), file=ckp.log_file)

    def __call__(self, *args):
        return self.model(*args)

    def get_model(self):
        return self.model
    
    def train(self):
        self.model.set_train(True)
    
    def eval(self):
        self.model.set_train(False)

    def state_dict(self, **kwargs):
        target = self.get_model()
        model_dict={}
        for p in target.get_parameters():
            model_dict[p.name]=p
        return model_dict

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        ms.save_checkpoint(
            target,
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            ms.save_checkpoint(
                target,
                os.path.join(apath, 'model', 'model_best.pt')
            )
        if self.save_middle_models:
            if epoch % 1 == 0:
                ms.save_checkpoint(
                    target,
                    os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
                )

    def load(self, apath, pre_train='.', resume=False, cpu=False, args=None):  #
        if pre_train != '.':
            if args.model == "TIME_PRIOR_PREDICT_WEIGHTED":
                print('Loading model from {}'.format(pre_train))
                target=self.get_model()
                ms.load_param_into_net(
                    target,
                    ms.load_checkpoint(pre_train)
                )
            elif args.model == '.':  #"VIDUE_WORSU"
                print('Loading model from {}'.format(pre_train))
                print("Excluding mismatching params....")
                pretrained_model = ms.load_checkpoint(pre_train)
                pretrained_dict = pretrained_model

                new_state_dict = pretrained_dict

                model_dict = self.model.state_dict()

                pretrained_dict = {k: v for k, v in new_state_dict.items() if
                                   k in model_dict and v.shape == model_dict[k].shape}

                model_dict.update(pretrained_dict)
                new_state_dict = model_dict

                target=self.get_model()
                ms.load_param_into_net(target, new_state_dict)
            else:
                print('Loading model from {}'.format(pre_train))
                print("Excluding time prior predictor....")
                pretrained_model = ms.load_checkpoint(pre_train)
                pretrained_dict = pretrained_model
                new_state_dict = pretrained_dict

                model_dict = self.model.state_dict()
                
                for name, param in new_state_dict.items():

                    if 'extractor' not in name and model_dict[name].shape == param.shape:   #and 'mlp' not in name
                        model_dict[name].copy(param)
                    else:
                        print('Not in name or mismatching', name)

                new_state_dict = model_dict

                target=self.get_model()
                ms.load_param_into_net(target, new_state_dict)

        elif resume:
            print('Loading model from {}'.format(os.path.join(apath, 'model', 'model_latest.pt')))
            target=self.get_model()
            ms.load_param_into_net(
                target,
                ms.load_checkpoint(os.path.join(apath, 'model', 'model_latest.pt')),
            )
        elif self.args.test_only:
            target=self.get_model()
            ms.load_param_into_net(
                target,
                ms.load_checkpoint(os.path.join(apath, 'model', 'model_latest.pt')),
            )
        else:
            pass
