'''
@author: Jingbo Lin
@contact: ljbxd180612@gmail.com
@github: github.com/mrluin
'''
import torch
import visdom
from visdom import Visdom

class visdomer(object):
    def __init__(self, port, hostname, model_name, compare_phase,
                 init_elements, init_params):
        self.port = port
        self.hostname = hostname
        self.env_name = model_name
        self.viz = Visdom(server=hostname, port=port, env=model_name)
        assert self.viz.check_connection(timeout_seconds=3), 'No connection could be formed quickly'
        self.compare_phase = compare_phase
        self.init_params = init_params

        self.windows = {}
        for element in init_elements:
            self.windows[element] = self.visdom_init(element)


    def get_viz(self):
        return self.viz

    def get_window(self, metric):
        # metric :: loss, accuracy,
        return self.windows[metric]

    def visdom_init(self, element):
        if element == 'lr':
            init_lr  = self.init_params.get('lr')
            assert init_lr is not None, 'init_lr is None when init visom of learning rate'
            window = self.viz.line(
                X = torch.ones(1),
                Y = torch.tensor([init_lr]),
                opts = dict(title = '{}'.format(element),
                            showlegend=True, legend=['{}'.format(element)],
                            xtype='linear', xlabel='epoch', xtickmin=0, xtick=True,
                            xtickstep=10, ytype='linear', ytickmin=0, ylabel='{}'.format(element),
                            ytick=True))
            return window
        elif element in ['loss', 'accuracy', 'miou', 'f1score']:
            assert isinstance(self.compare_phase, list) and len(self.compare_phase) == 2, 'compare_phase must be list and length with 2'
            window = self.viz.line(
                X=torch.stack((torch.ones(1), torch.ones(1)), 1),
                Y=torch.stack((torch.ones(1), torch.ones(1)), 1),
                opts=dict(title='{}-{}-{}'.format(self.compare_phase[0], self.compare_phase[1], element),
                          showlegend=True,
                          legend=['{}-{}'.format(self.compare_phase[0], element),
                                  '{}-{}'.format(self.compare_phase[1], element)],
                          xtype='linear', label='epoch', xtickmin=0,
                          xtick=True, xtickstep=10, ytype='linear',
                          ylabel='{}'.format(element), ytickmin=0, ytick=True))
        else: raise NotImplementedError('do not support metric {}'.format(element))
        return window

    def visdom_update(self, epoch, update_element, update_value):
        if update_element in ['loss', 'accuracy', 'miou', 'f1score']:
            #print(update_value)
            assert isinstance(update_value, list) and len(update_value) == 2, 'update_value should be list and with length 2, but got {:} with length {:}'.format(type(update_value), len(update_value))
            train_log = update_value[0]
            valid_log = update_value[1]
            window = self.get_window(update_element)
            self.viz.line(
                X = torch.stack((torch.ones(1) * epoch, torch.ones(1) * epoch), 1),
                Y = torch.stack((torch.tensor([train_log]), torch.tensor([valid_log])), 1),
                win = window,
                update='append' if epoch != 1 else 'insert'
            )
        elif update_element == 'lr':
            current_lr = update_value[0]
            window = self.get_window(update_element)
            self.viz.line(
                X = torch.ones(1) * epoch,
                Y = torch.tensor([current_lr]),
                win = window,
                update = 'append' if epoch != 1 else 'insert'
            )
