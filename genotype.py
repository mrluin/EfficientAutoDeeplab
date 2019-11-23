from collections import namedtuple

from copy import deepcopy

Genotype = namedtuple('Genotype', 'cell cell_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

class Structure:
    def __init__(self, genotype):
        assert isinstance(genotype, list) or isinstance(genotype, tuple), 'invalid class of genotype :{:}'.format(type(genotype))
        self.node_num = len(genotype) + 1
        self.nodes = []
        self.node_N = []
        for index, node_info in enumerate(genotype): # index, node index
            assert isinstance(node_info, list) or isinstance(node_info, tuple), 'invalid class of node_info:{:}'.format(type(node_info))
            assert len(node_info) >= 1, 'invalid length: {:}'.format(len(node_info))
            for node_in in node_info:
                assert isinstance(node_in, list) or isinstance(node_in, tuple), 'invalid class of node_in:{:}'.format(type(node_in))
                assert len(node_in) == 2 and node_in[1] <= index, 'invalid node_in: {}'.format(node_in)
            self.node_N.append(len(node_info))
            self.nodes.append(tuple(deepcopy(node_info)))
    def tolist(self, remove_str):
        # convert this class to the list, remove_str is none, then remove 'none' operation
        # re-order the input node in this function
        # return genotype-list and success
        genotypes = []
        for node_info in self.nodes: # node_info [op_name, start_index_j]
            node_info = list(node_info)
            node_info = sorted(node_info, key=lambda x: (x[1], x[0]))
            # sorted by start_index_j
            node_info = tuple(filter(lambda x: x[0] != remove_str, node_info))
            if len(node_info) == 0: return None, False
            genotypes.append(node_info)
        return genotypes, True

    def node(self, index):
        assert index > 0 and index <= len(self), 'invalid index={:} < {:}'.format(index, len(self))
        return self.nodes[index]

    def tostr(self):
        strings = []
        for node_info in self.nodes:
            string = '|'.join(x[0]+'~{:}'.format(x[1]) for x in node_info)
            string = '|{:}|'.format(string)
            strings.append(string)
        return '+'.join(strings)

    def check_valid(self):
        # TODO: for what?
        nodes = {0:True}
        for i, node_info in enumerate(self.nodes):
            sums = []
            for op, xin in node_info:
                if op == 'none' or nodes[xin] == False: x=False
                else: x = True
                sums.append(x)
            nodes[i+1] = sum(sums) > 0
        return nodes[len(self.nodes)]

    def to_unique_str(self):

        nodes = {0:'0'}
        for i_node, node_info in enumerate(self.nodes):
            cur_node = []
            for op, xin in node_info:
                if op == 'skip_connect': x = nodes[xin]
                else: x = '('+nodes[xin]+')'+'@{:}'.format(op)
                cur_node.append(x)

            nodes[i_node+1] = '+'.join(sorted(cur_node))
        return nodes[len(self.nodes)]

    def check_valid_op(self, op_name):
        for node_info in self.nodes:
            for inode_edge in node_info:
                if inode_edge[0] not in op_name: return False
        return True

    def __repr__(self):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, item):
        raise NotImplementedError
    '''
    @staticmethod
    def str2structure(xstr):

    @staticmethod
    def str2fullstructure(xstr, default_name='none'):

    @staticmethod
    def gen_all(search_space, num, return_ori):
        assert isinstance(search_space, list) or \
               isinstance(search_space, tuple), 'invalid class of search-space : {:}'.format(type(search_space))
        assert num >= 2, 'There should be at least two nodes in a neural cell instead of {:}'.format(num)
        all_archs = get_combination(search_space, 1)
        for i, arch in enumerate(all_archs):
            all_archs[i] = [tuple(arch)]

        for inode in range(2, num):
            cur_nodes = get_com
    '''