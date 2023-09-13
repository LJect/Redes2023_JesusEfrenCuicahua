"""
network1.py
~~~~~~~~~~
red neuronal para clasificar digitos, usando el algoritmo SGD y una funcion de activacion sigmoide
"""

#### importamoslibrerias
import random
import numpy as np
#hacemos la calse para crear la red 
class Network(object):
#funcion para crear la red con el tamaño indicado en la lista sizes, donde el numero de elementos indica el numero de capas 
#que tendra la red y el valor del elemento el numero de neuronas en la capa  
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        """iniciamos los valores de los pesos y bias de forma aleatoria con numero entre 0 y 1 estos se guardan en 
        listas donde los elementos son listas de los pesos y bias de cada capa en orden """
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]    #np.random.randn(a,b) crra una lista con a filas y b
        self.weights = [np.random.randn(y, x)                      # columnas con numero aleatorios
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.Pw = [np.zeros(w.shape) for w in self.weights]  #creamo doslistas par almacenar los P's de los correspondietes...
        self.Pb=[np.zeros(b.shape) for b in self.biases]   # b's y w's para ajustar sus pesos, los iniciamos en cero
        
#Funcion que recibe un elemento de entrada 'x' y lo asiga a 'a' para calcular la salida de la red, regresa el valor de la
#salida de la red 'a' (vector)
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b) #llama a la funcion sigmoide y le pasa los argumentos de pesos, bias y avtivaciones
        #a=np.exp(a)/(np.sum(np.exp(a))) #aplicamos la funcion soft max a la salida de la red 
        return a    #de capas anteriores.
#funcion que aplica el algoritmo SGD para entrenar la red y ademas si se le asignan datos de prueba, 
#la red testea esos datos y nos devuelve el numero de aciertos que obtuvo.
    def SGD(self, training_data, epochs, mini_batch_size, eta, beta, 
            test_data=None):
        if test_data: n_test = len(test_data)   #si hay datos de entrenamiento determina el numero de datos
        n = len(training_data)                  #determina el numero de datos de entrenamiento 
        """inicia con el entrenamiento para las epocas que le hemos idicado"""
        for j in range(epochs):         
            random.shuffle(training_data)   #mescla los datos de entrenamiento con la funcion random.shuffle
            """ forma los mini-batches de acuerdo al numero que le indicamos, y es una lista(mini_batches)
            que son los subconjuntos en que se dividio el total de datos de entrenamiento """
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            """comienza el entrenamiento de la red para cad mini batch en mini_batches usa la funcion update_mini_batch
            y le pasa uno por uno cada minibatch y el eta"""
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, beta)    
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(       #si hay datos de prueba, los introduce e imprime 
                    j, self.evaluate(test_data), n_test))   # el total de acierto, y el total de datos.
            else:
                print ("Epoch {0} complete".format(j))
#funcion que actualiza los pesos y bias al termino de cada minibatch usando el SGD
    def update_mini_batch(self, mini_batch, eta, beta):
        """crea listas que contiene como elementos arrays de ceros de tamaño de los bias y pesos segun la capa
        estas listans forma el gradiente de c pues cada entrada elemento dentro de los array representa una parcial 
        de c respeto a el 'b' o 'w' correspondiente"""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #np.zeros crea arreglos de ceros de la forma especificada
        nabla_w = [np.zeros(w.shape) for w in self.weights]#A.shape da la forma deñ elemento antes del punto (A)  
        """utiliza el algoritmo backpropagation para calcular los elementos de las listas nabla para cada elemento
        de el minibatch y los suma con los ya obtenidos"""
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #suma de los componentes del gradiente para cada
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #elemento del minibatch 
        """actualizacion de los pesos una vez calculado el gradinate con los datos del minibatch y sumamos el P para
        darle la inercia en la actualizacion de los pesos, le sumamos los P's(t-1) multiplicados por eta."""
        self.weights = [w-(eta/len(mini_batch))*nw+(eta*pw)
                        for w, nw, pw in zip(self.weights, nabla_w, self.Pw)]
        self.biases = [b-(eta/len(mini_batch))*nb+(eta*pb)
                       for b, nb, pb in zip(self.biases, nabla_b, self.Pb)]
        self.Pb=[-b+(beta*pb) for b, pb in zip(nabla_b, self.Pb)] #actualizamos los P's con el gradiente y los P's(t-1)
        self.Pw=[-w+(beta*pw) for w, pw in zip(nabla_w, self.Pw)] #para usarlos en la ssiguiente actualización:
#funcion del agoritmo backpropagation que retorna dos listas, con las parciales respecto a los b's y w's que 
#que conforman el gradiente de la funcion de costo par aun elemento de los datos de entrenamiento
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward para calcular las activaciones correspondientes a cada capa y calcular las parciales 
        activation = x  #datos de entrada de la primera capa
        activations = [x] # se guardan las activaciones de cada capa 
        zs = [] # lista de los z's en cada capa 
        """calculamos las activaciones de todas las neuronas para guardarlos en una lista que contiene como 
        elementos las activaciones de todas las neruonas en cada capa."""
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b     #forma los z
            zs.append(z)    #agrega los z de cada capa a la lista de los z's
            activation = sigmoid(z) #aplica la funcio sigmoide al vector z actual para calcular la activacion de la neurona
            activations.append(activation) #guarda el vector obtenido en la lista de las activaciones por capa.
        # backward pass para poder calcular los delta correspondietes, comenzando con la ultima capa
        """ calcula el delta de la ultima capa con la funcio cost.derivative a la que le pasa como argumento 
        las activaciones de la ultima capa, osea el ultimo elemento de la lista activationn y la salida correspondiente
        a este dato de entrenamiento, y el retorno de la funcion lo multiplica por la derivada de la funcio sigmoide 
        evaluada enlos z's de la ultima capa teniendo asi las delta de la ultima capa """
        #delta = self.cost_derivative(activations[-1], y) * \
        #   sigmoid_prime(zs[-1])
        #activations[-1]=softmax(activations[-1])# aplicamos la funcio softmax a las salidas de la ultima capa.
        delta = activations[-1]- y  #redefinimos la delta de la ultima capa pues aqui es donde se ve el cambio para
                                    #la funcion de costo cross entropy y la funcio softmax.  
        nabla_b[-1] = delta #actualiza la parcial de la fucnio de costo respecto a los b's de la ultima capa.
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #actualiza parcial de c respecto a los w's de la ultima capa.
        #calculamos los deltas y asi las parciales de las capas anteriores, pues ya contamos con el 
        # delta dela ultima capa.
        for l in range(2, self.num_layers):
            z = zs[-l]  #asignamos los z de las capas a este elemento, comenzando con los de la penultima capa y en cada
                        # iteracion lo cambiamos por lo de la capa anterior.
            sp = sigmoid_prime(z) #evaluamos la derivada de la funcio sigmoide con los z asignados en el paso anterior.
            """calculamos los delta de la capa correspondiente en cada iteracion, esto se hace realizando el producto
            matricial de la matriz trnaspuesta de los pesos en la capa siguiente y los delta de la capa siguiente, para
            despues multiplicarlos por el resultado sp que se menciono antes."""
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp  
            """calculamos las parciales de c respecto a cada w y b de las capas y los sustituimos en los vectores
            nabla_b y nabla_w que inicialmente tenia ceros"""
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)   #regresamos los componentes del gradiante de C para el dato proporcionado 
# fucion que le da a la red datos de prueba y evalua cuantos aciertos obtiene la red, 
    def evaluate(self, test_data):
        """ determina cual es la posicion de elemento mas alto de la salida de feeforward con la funcion np.argmax
        y dado que cuenta co 10 posiciones se puede tomar como que cada indice corresponde justamente al numero que 
        predice la red"""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)#compara los resultados y suma los aciertos de la red
    #esta funcion nos retorna la derivada de Cx con respecto la activacion a (a-y) para la ultima capa.
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

#funcion de activacion sigmoide, retorna el vector de evaluacion de la funcion con el vector z proporcionado 
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
#derivada de la funcion sigmoide, regresa el vector de evaluar la derivada en cada componente del vecto z dado 
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
#funcion softmax
def softmax(z):
    "aplicamos la funcion softmax a las salidas de la red "
    a=np.exp(z)
    s=np.sum(a)
    v=a/s
    #print(z,'l')
    return v
