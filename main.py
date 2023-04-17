from multiprocessing import Process
from dnn import Salida
#from DNN.captura import Captura


def runInParallel(*fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        #print("PROCESADORES", p)
        p.start()
        proc.append(p)
    for p in proc:
        #print("SALIDA PROCESADORES", p)
        p.join()


if __name__ == '__main__':
    #runInParallel(Captura(CUDA = True))
    runInParallel(Salida(CUDA=True))
