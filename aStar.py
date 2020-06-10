import cv2
import sys
import timeit
import numpy as np
from PIL import Image


class Node():
    # resimden alınan pixelleri temsil edecek node
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.created = False

        self.g = sys.maxsize
        self.h = sys.maxsize
        self.f = sys.maxsize

    def __eq__(self, other):
        return self.position == other.position

class MinHeap:
    def __init__(self):
        super().__init__()
        temp_node = Node(None, None)
        self.heap = [temp_node]

    def push(self, data): # heap'e eleman ekleyen fonksiyon
        self.heap.append(data)
        self.__floatUp(len(self.heap) - 1)


    def pop(self): # heap'den elaman çıkaran fonksiyon
        if len(self.heap) > 2:
            self.__swap(1, len(self.heap) - 1)
            minimum = self.heap.pop()
            self.bubbleDown(1)
        elif len(self.heap) == 2:
            minimum = self.heap.pop()
        else:
            minimum = False
        return minimum

    def __swap(self, i, j): # iki elemanın yerini degistiren fonksiyon
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def __floatUp(self, index): # elamanın ağaçta dogru yere yerleşmesini sağlayan fonksiyon
        parent = index // 2
        if index <= 1:
            return
        elif self.heap[index].f < self.heap[parent].f:
            self.__swap(index, parent)
            self.__floatUp(parent)

    def bubbleDown(self, index): # min heapify
        left = index * 2
        right = index * 2 + 1
        largest = index
        if len(self.heap) > left and self.heap[largest].f > self.heap[left].f:
            largest = left
        if len(self.heap) > right and self.heap[largest].f > self.heap[right].f:
            largest = right
        if largest != index:
            self.__swap(index, largest)
            self.bubbleDown(largest)

def find_path(current_node,img): # iki nokta arasındaki yolu çıkaran ve yeşil yapan fonksiyon
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        img[current.position[0]][current.position[1]] = [0, 255, 0]
        current = current.parent
    return path[::-1], img  



def min_stack(open_list): # stackteki minimum elemanı bulan fonksiyon
    current_node = open_list[0]
    current_index = 0

    for index, item in enumerate(open_list):
        if item.f < current_node.f:
            current_node = item
            current_index = index
    return current_node, current_index, open_list


def aStar_with_stack(start, end, img): # a* / stack algoritması, başlangıc bitiş ve fotoğraf parametreleri alır

    start_node = Node(None, start) # başlangıç noktası
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end) #  bitiş noktası
    end_node.g = end_node.h = end_node.f = 0

    # algoritma için gerekli closed list ve open list tanımlaması
    closed_list = []
    open_list = []
    pop_count = 0 # stackteki işlemleri tutabilmek için tanımlanan değişken
    stack_max_count = 1 # stackteki max eleman sayısını bulmak için tanımlanan değişken
    open_list.append(start_node)

    created = np.zeros((img.shape[0], img.shape[1])) # bütün noktaların oluşup oluşmadığını tutan matris( 0 : oluşmadı , 1 : open_list'te , 2: closed_list'te)
    created[start_node.position[0]][start_node.position[1]] = 1

    while len(open_list) > 0: # open list boşalana kadar döngü
        if len(open_list) > stack_max_count: # max_count'ı belirlemek için
            stack_max_count = len(open_list) # max_count'ı belirlemek için
    
        # stackteki min eleman belirlenir
        current_node, current_index, open_list = min_stack(open_list)

        # open_list'teki eleman poplanır ve closde list'e konur
        open_list.pop(current_index)
        pop_count += 1
        closed_list.append(current_node)
        created[current_node.position[0]][current_node.position[1]] = 2 #created güncellenir

        # hedef bulunduysa yol yazdırılır
        if current_node == end_node:
            print("stack islem sayisi:"+ str(pop_count))
            print("stack max eleman sayisi:"+ str(stack_max_count))
            path, img = find_path(current_node,img)
            return path, img

        # komsular belirlenir
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # komsu noktalar

            # node konumu
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # node konumunun size içinde olduğu kontrol edilir
            if node_position[0] > (len(img) - 1) or node_position[0] < 0 or node_position[1] > (len(img[len(img)-1]) -1) or node_position[1] < 0:
                continue

            new_node = Node(current_node, node_position)
            children.append(new_node)

        # komsular dolaşılır
        for child in children:
            if created[child.position[0]][child.position[1]] == 2: # 2 ise closed listte, bir şey yapılmasına gerek yok
                continue

            if created[child.position[0]][child.position[1]] == 1: # 1 ise open listte, open listte komşu bulunur
                i = 0
                while i < len(open_list):
                    if child == open_list[i]:
                        child = open_list[i]
                        i = len(open_list)
                    i += 1

            new_path_cost = current_node.g + (256 - img[child.position[0]][child.position[1]][2])/100 # yeni maliyet hesaplanır

            if new_path_cost < child.g: # hesaplanan maliyet varolan maliyetten küçükse güncellenir
                child.g = new_path_cost

                x, y = abs(child.position[0] - end_node.position[0]), abs(child.position[1] - end_node.position[1]) # x ve y farkları hesaplanır
                child.h = x+y   # x ve y farkları toplandı
                child.f = child.g + child.h     # f fonksiyonu hesaplandı
                child.parent = current_node

            # komsu daha önce oluşturulmamışsa open liste eklenir
            if created[child.position[0]][child.position[1]] == 0:
                open_list.append(child)
                created[child.position[0]][child.position[1]] =1


def aStar_with_heap(start,end,img): # a* / heap algoritması
    """ bu algoritmadaki çoğu adım stack versiyonuna benzer"""

    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    closed_list = []
    minheap = MinHeap()
    minheap.push(start_node)

    created = np.zeros((img.shape[0], img.shape[1]))
    created[start_node.position[0]][start_node.position[1]] = 1

    while len(minheap.heap) > 1:

        current_node = minheap.pop()
        closed_list.append(current_node)
        created[current_node.position[0]][current_node.position[1]] = 2

        if current_node == end_node:
            path, img = find_path(current_node, img)
            return path, img

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  

            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if node_position[0] > (len(img) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(img[len(img) - 1]) - 1) or node_position[1] < 0:
                continue

            new_node = Node(current_node, node_position)

            children.append(new_node)

        for child in children:
            if created[child.position[0]][child.position[1]] == 2:
                continue

            if created[child.position[0]][child.position[1]] == 1:
                i = 0
                while i < len(minheap.heap):
                    if child == minheap.heap[i]:
                        child = minheap.heap[i]
                        i = len(minheap.heap)
                    i += 1

            new_path_cost = current_node.g + (256 - img[child.position[0]][child.position[1]][2])/100

            if new_path_cost < child.g:
                child.g = new_path_cost

                x, y = abs(child.position[0] - end_node.position[0]), abs(child.position[1] - end_node.position[1])
                child.h = x+y
                child.f = child.g + child.h
                child.parent = current_node


            if created[child.position[0]][child.position[1]] == 0:

                minheap.push(child)
                created[child.position[0]][child.position[1]] =1


def bfs_with_stack(start, end, img): # best first search / stack algoritması

    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0


    closed_list = []
    open_list = []
    pop_count = 0
    stack_max_count = 1
    open_list.append(start_node)

    created = np.zeros((img.shape[0],img.shape[1]))
    created[start_node.position[0]][start_node.position[1]] = 1


    while len(open_list) > 0:
        if len(open_list) > stack_max_count:
            stack_max_count = len(open_list)

        current_node, current_index, open_list = min_stack(open_list)

        open_list.pop(current_index)
        pop_count += 1
        closed_list.append(current_node)
        created[current_node.position[0]][current_node.position[1]] = 2

        if current_node == end_node:
            print("stack islem sayisi:" + str(pop_count))
            print("stack max eleman sayisi:" + str(stack_max_count))
            path, img = find_path(current_node, img)
            return path, img

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: 

            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if node_position[0] > (len(img) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(img[len(img) - 1]) - 1) or node_position[1] < 0:
                continue

            new_node = Node(current_node, node_position)

            children.append(new_node)

        for child in children:
            
            if created[child.position[0]][child.position[1]] == 1 or created[child.position[0]][child.position[1]] == 2:
                continue

            x, y = abs(child.position[0] - end_node.position[0]), abs(child.position[1] - end_node.position[1])
            child.h = x+y

            child.f = child.h
            child.parent = current_node

            open_list.append(child)
            created[child.position[0]][child.position[1]] = 1


def bfs_with_heap(start, end, img): # best first search / heap algoritması

    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    closed_list = []
    created = np.zeros((img.shape[0], img.shape[1]))
    created[start_node.position[0]][start_node.position[1]] = 1

    minheap = MinHeap()
    minheap.push(start_node)

    while len(minheap.heap) > 1:

        current_node = minheap.pop()

        closed_list.append(current_node)
        created[current_node.position[0]][current_node.position[1]] = 2

        if current_node == end_node:
            path, img = find_path(current_node, img)
            return path, img

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  

            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if node_position[0] > (len(img) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(img[len(img) - 1]) - 1) or node_position[1] < 0:
                continue

            new_node = Node(current_node, node_position)

            children.append(new_node)

        for child in children:

            if created[child.position[0]][child.position[1]] == 1 or created[child.position[0]][child.position[1]] == 2:
                continue

            x, y = abs(child.position[0] - end_node.position[0]), abs(child.position[1] - end_node.position[1])
            child.h = x + y

            child.f = child.h
            child.parent = current_node

            minheap.push(child)
            created[child.position[0]][child.position[1]] = 1


def main():

    run = True

    while run:
        img_name = input("Input resmin adini girin (cikmak icin exit yazabilirsiniz):")
        if img_name == "exit":
            run = False

        else:

            start_x, start_y, end_x, end_y = sys.maxsize,sys.maxsize,sys.maxsize,sys.maxsize
            img = cv2.imread(img_name)  # B G R , for R value, use with [2]

            while start_x > img.shape[0]-1 or start_y > img.shape[1]-1 or end_x > img.shape[0]-1 or end_y > img.shape[1]-1:
                print("Baslangic bitis noktasi koordinatlari (0,0) , ("+str(img.shape[0]-1)+","+str(img.shape[1]-1)+") arasi olmalidir")

                start_x = int(input("Baslangic noktasi icin x girin:"))
                start_y = int(input("Baslangic noktasi icin y girin:"))


                end_x = int(input("Bitis noktasi icin x girin:"))
                end_y = int(input("Bitis noktasi icin y girin:"))

            start = (start_x, start_y)
            end = (end_x, end_y)
            algorithm_choice = 0

            while algorithm_choice < 1 or algorithm_choice > 4:
                print("1. Astar-Stack")
                print("2. Astar-Heap")
                print("3. BFS-Stack")
                print("4. BFS-Heap")
                print("Algoritma secin:", end="")
                algorithm_choice = int(input())

            if algorithm_choice == 1:
                begin_time = timeit.default_timer()
                path, img = aStar_with_stack(start, end, img)
                stop_time = timeit.default_timer()

            elif algorithm_choice == 2:
                begin_time = timeit.default_timer()
                path, img = aStar_with_heap(start, end, img)
                stop_time = timeit.default_timer()

            elif algorithm_choice == 3:
                begin_time = timeit.default_timer()
                path, img = bfs_with_stack(start, end, img)
                stop_time = timeit.default_timer()

            else:
                begin_time = timeit.default_timer()
                path, img = bfs_with_heap(start, end, img)
                stop_time = timeit.default_timer()

            print(path)
            print('Time: ', stop_time - begin_time)
            cv2.imwrite("sonuc.jpg", img)
            image = Image.open('sonuc.jpg')
            image.show()


if __name__ == '__main__':
    main()