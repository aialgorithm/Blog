"""
Created on ：2019/04/7
@author: Freeman, feverfc1994
"""

from PIL import Image
import pydotplus as pdp
from GBDT.decision_tree import Node, Tree
import os
import matplotlib.pyplot as plt


def plot_multi(trees: dict, max_depth: int, iter: int):
    trees_traversal = {}
    trees_nodes = {}
    for class_index in trees.keys():
        tree = trees[class_index]
        res = []
        root = tree.root_node
        traversal(root,res)
        trees_traversal[class_index] = res
        # 获取所有节点
        nodes = {}
        index = 0
        for i in res:
            p, c = i[0], i[1]
            if p not in nodes.values():
                nodes[index] = p
                index = index + 1
            if c not in nodes.values():
                nodes[index] = c
                index = index + 1
        trees_nodes[class_index] = nodes
        # 通过dot语法将决策树展示出来
    trees_edges = {}
    trees_node = {}
    for class_index in trees.keys():
        trees_node[class_index] = ''
        trees_edges[class_index] = ''
    for depth in range(max_depth):
        for class_index in trees.keys():
            for nodepair in trees_traversal[class_index]:
                if nodepair[0].deep == depth:
                    p, c = nodepair[0], nodepair[1]
                    l = len([i for i in range(len(c.data_index)) if c.data_index[i] is True])
                    pname = str(list(trees_nodes[class_index].keys())[list(trees_nodes[class_index].values()).index(p)])
                    cname = str(list(trees_nodes[class_index].keys())[list(trees_nodes[class_index].values()).index(c)])
                    if l > 0:
                        trees_edges[class_index] = trees_edges[class_index] + pname + '->' + cname + '[label=\"' + str(p.split_feature) + (
                            '<' if p.left_child == c else '>=') + str(p.split_value) + '\"]' + ';\n'

                        trees_node[class_index] = trees_node[class_index] + pname + '[width=1,height=0.5,color=lemonchiffon,style=filled,shape=ellipse,label=\"id:' + str(
                        [i for i in range(len(p.data_index)) if p.data_index[i] is True]) + '\"];\n' + \
                           (
                               cname + '[width=1,height=0.5,color=lemonchiffon,style=filled,shape=ellipse,label=\"id:' + str(
                                   [i for i in range(len(c.data_index)) if
                                    c.data_index[i] is True]) + '\"];\n' if l > 0 else '')
                    if c.is_leaf and l > 0:
                        trees_edges[class_index] = trees_edges[class_index] + cname + '->' + cname + 'p[style=dotted];\n'
                        trees_node[class_index] = trees_node[class_index] + cname + 'p[width=1,height=0.5,color=lightskyblue,style=filled,shape=box,label=\"' + str(
                            "{:.4f}".format(c.predict_value)) + '\"];\n'
                else:
                    continue
            dot = '''digraph g {\n''' + trees_edges[class_index] + trees_node[class_index] + '''}'''
            graph = pdp.graph_from_dot_data(dot)
            # 保存图片+pyplot展示
            graph.write_png('results/NO.{}_{}_tree.png'.format(iter, class_index))
        plt.ion()
        plt.figure(1, figsize=(30, 20))
        plt.axis('off')
        plt.title('NO.{} iter '.format(iter))
        class_num = len(trees.keys())
        if class_num / 3 - int(class_num / 3) <0.000001:
            rows = int(class_num/3)
        else:
            rows = int(class_num/3)+1
        for class_index in trees.keys():
            index = list(trees.keys()).index(class_index)
            plt.subplot(rows, 3, index+1)
            img = Image.open('results/NO.{}_{}_tree.png'.format(iter, class_index))
            img = img.resize((1024, 700), Image.ANTIALIAS)
            plt.axis('off')
            plt.title('NO.{}_class {}'.format(iter, class_index))
            plt.rcParams['figure.figsize'] = (30.0, 20.0)
            plt.imshow(img)
        plt.savefig('results/NO.{}_tree.png'.format(iter))
        plt.pause(0.01)


def plot_tree(tree: Tree, max_depth: int, iter: int):
    """
           展示单棵决策树
    :param tree: 生成的决策树
    :param max_depth: 决策树的最大深度
    :param iter: 第几棵决策树
    :return:
    """
    root = tree.root_node
    res = []
    # 通过遍历获取决策树的父子节点关系，可选有traversal 层次遍历 和traversal_preorder 先序遍历
    traversal(root, res)

    # 获取所有节点
    nodes = {}
    index = 0
    for i in res:
        p, c = i[0], i[1]
        if p not in nodes.values():
            nodes[index] = p
            index = index + 1
        if c not in nodes.values():
            nodes[index] = c
            index = index + 1

    # 通过dot语法将决策树展示出来
    edges = ''
    node = ''
    # 将节点层次展示
    for depth in range(max_depth):
        for nodepair in res:
            if nodepair[0].deep == depth:
                # p,c分别为节点对中的父节点和子节点
                p, c = nodepair[0], nodepair[1]
                l = len([i for i in range(len(c.data_index)) if c.data_index[i] is True])
                pname = str(list(nodes.keys())[list(nodes.values()).index(p)])
                cname = str(list(nodes.keys())[list(nodes.values()).index(c)])
                if l > 0:
                    edges = edges + pname + '->' + cname + '[label=\"' + str(p.split_feature) + (
                        '<' if p.left_child == c else '>=') + str(p.split_value) + '\"]' + ';\n'

                node = node + pname + '[width=1,height=0.5,color=lemonchiffon,style=filled,shape=ellipse,label=\"id:' + str(
                    [i for i in range(len(p.data_index)) if p.data_index[i] is True]) + '\"];\n' + \
                       (cname + '[width=1,height=0.5,color=lemonchiffon,style=filled,shape=ellipse,label=\"id:' + str(
                    [i for i in range(len(c.data_index)) if c.data_index[i] is True]) + '\"];\n' if l > 0 else '')
                if c.is_leaf and l > 0:
                    edges = edges + cname + '->' + cname + 'p[style=dotted];\n'
                    node = node + cname + 'p[width=1,height=0.5,color=lightskyblue,style=filled,shape=box,label=\"' + str(
                        "{:.4f}".format(c.predict_value)) + '\"];\n'
            else:
                continue
        dot = '''digraph g {\n''' + edges + node + '''}'''
        graph = pdp.graph_from_dot_data(dot)
        # 保存图片+pyplot展示
        graph.write_png('results/NO.{}_tree.png'.format(iter))
        img = Image.open('results/NO.{}_tree.png'.format(iter))
        img = img.resize((1024, 700), Image.ANTIALIAS)
        plt.ion()
        plt.figure(1, figsize=(30, 20))
        plt.axis('off')
        plt.title('NO.{} tree'.format(iter))
        plt.rcParams['figure.figsize'] = (30.0, 20.0)
        plt.imshow(img)
        plt.pause(0.01)


def plot_all_trees(numberOfTrees: int):
    '''
           将所有生成的决策树集中到一张图中展示
    :param numberOfTrees: 决策树的数量
    :return:
    '''
    # 每行展示3棵决策树 根据决策树数量决定行数
    if numberOfTrees / 3 - int(numberOfTrees / 3) > 0.000001:
        rows = int(numberOfTrees / 3)+1
    else:
        rows = int(numberOfTrees / 3)
    # 利用subplot 将所有决策树在一个figure中展示
    plt.figure(1, figsize=(30,20))
    plt.axis('off')
    try:
        for index in range(1, numberOfTrees + 1):
            path = os.path.join('results', 'NO.{}_tree.png'.format(index))
            plt.subplot(rows, 3, index)
            img = Image.open(path)
            img = img.resize((1000, 800), Image.ANTIALIAS)
            plt.axis('off')
            plt.title('NO.{} tree'.format(index))
            plt.imshow(img)
        plt.savefig('results/all_trees.png', dpi=300)
        plt.show()
        # 由于pyplot图片像素不是很高，使用方法生成高质量的图片
        image_compose(numberOfTrees)
    except Exception as e:
        raise e


def image_compose(numberOfTrees: int):
    '''
           将numberOfTrees棵决策树的图片拼接到一张图片上
    :param numberOfTrees: 决策树的数量
    :return:
    '''

    png_to_compose = []
    # 获取每张图片的size
    for index in range(1,numberOfTrees+1):
        png_to_compose.append('NO.{}_tree.png'.format(index))
    try:
        path = os.path.join('results', png_to_compose[0])
        shape = Image.open(path).size
    except Exception as e:
        raise e
    IMAGE_WIDTH = shape[0]
    IMAGE_HEIGET = shape[1]
    IMAGE_COLUMN = 3

    if len(png_to_compose)/IMAGE_COLUMN - int(len(png_to_compose)/IMAGE_COLUMN) > 0.0000001:
        IMAGE_ROW = int(len(png_to_compose)/IMAGE_COLUMN)+1
    else:
        IMAGE_ROW = int(len(png_to_compose) / IMAGE_COLUMN)
    # 新建一张用于拼接的图片
    to_image = Image.new('RGB', (IMAGE_COLUMN*IMAGE_WIDTH, IMAGE_ROW*IMAGE_HEIGET), '#FFFFFF')
    # 拼接图片
    for y in  range(IMAGE_ROW):
        for x in range(IMAGE_COLUMN):
            if y*IMAGE_COLUMN+x+1 > len(png_to_compose):
                break
            path = os.path.join('results', 'NO.'+str(y*IMAGE_COLUMN+x+1)+'_tree.png')
            from_image = Image.open(path)
            to_image.paste(from_image, (x*IMAGE_WIDTH, y*IMAGE_HEIGET))

    to_image.save('results/all_trees_high_quality.png')


def traversal_preorder(root: Node, res: list):
    '''

          先序遍历决策树获取节点间的父子关系
    :param root: 决策树的根节点
    :param res:  存储节点对(父节点,子节点)的list
    :return: res
    '''
    if root is None:
        return
    if root.left_child is not None:
        res.append([root, root.left_child])
        traversal_preorder(root.left_child, res)
    if root.right_child is not None:
        res.append([root, root.right_child])
        traversal_preorder(root.right_child, res)


def traversal(root: Node, res: list):
    '''

              层次遍历决策树获取节点间的父子关系
        :param root: 决策树的根节点
        :param res:  存储节点对(父节点,子节点)的list
        :return: res
        '''
    outList = []
    queue = [root]
    while queue != [] and root:
        outList.append(queue[0].data_index)
        if queue[0].left_child != None:
            queue.append(queue[0].left_child)
            res.append([queue[0], queue[0].left_child])
        if queue[0].right_child != None:
            queue.append(queue[0].right_child)
            res.append([queue[0], queue[0].right_child])
        queue.pop(0)


if __name__ =="__main__":
    plot_all_trees(10)
    # image_compose(10)