#EJEMPLO 1:

#1-Definicion de una clase
class Persona:
    especie = 'Humano' # atributo de clase
    def __init__(self, nombre, edad):
        self.nombre = nombre # atributo de instancia
        self.edad = edad # atributo de instancia

    def saludar(self):
        print(f'Hola, me llamo {self.nombre} y tengo {self.edad} años')

#2-Creacion de objetos
juan = Persona('Juan', 30)
maria = Persona('Maria', 25)

juan.saludar() # Salida: Hola, me llamo Juan y tengo 30 años
maria.saludar() # Salida: Hola, me llamo Maria y tengo 25 años\

carlos = Persona('Juan', 30)
print(carlos.especie) # Salida: Humano



#EJEMPLO 2:
#ENCAPSULAMIENTO
#el _ otorga el nivel de protegido

class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre  # Público
        self._profesion = "Desconocida"  # Protegido
        self.__saldo = 1000  # Privado

    def mostrar_saldo(self):
        print(f"Saldo: {self.__saldo}")

juan = Persona("Juan", 30)
print(juan.nombre)  # Accesible
print(juan._profesion)  # Accesible, pero no recomendado
# print(juan.__saldo)  # Error
juan.mostrar_saldo()  # Salida: Saldo: 1000



#EJEMPLO 3:
#HERENCIA

class Persona:
    def __init__(self, nombre):
        self.nombre = nombre
    
    def saludar(self):
        print(f'Hola, soy {self.nombre}')

#clase hija
class Estudiante(Persona):
    def __init__(self,nombre, carrera):
        super().__init__(nombre) #llama al constructor de la clase padre
        self.carrera = carrera
    
    def estudiar(self):
        print(f'{self.nombre} esta estudiando {self.carrera}')

marco = Estudiante('Marco', 'Quimica')

marco.saludar() # Salida: Hola, soy Marco
marco.estudiar() # Salida: Marco esta estudiando Quimica



#EJEMPLO 4:
#POLIMORFISMO

class Animal:
    def hablar(self):
        pass

class Perro(Animal):
    def hablar(self):
        print('Guau!')

class Gato(Animal):
    def hablar(self):
        print('Miau!')

animales = [Perro(), Gato()]

for animal in animales:
    print(animal.hablar()) # Salida: Guau! \n Miau!


#Existen metodos especiales(__init__,__str__,__len__,__add__,__sub__,__mul__,__truediv__,__floordiv__,__mod__,__pow__,__eq__,__ne__,__lt__,__gt__,__le__,__ge__)
#__init__ : Constructor de la clase
#__str__ : Devuelve una representación en forma de cadena del objeto
#__len__ : Devuelve la longitud del objeto
#__add__ : Suma de objetos
#__sub__ : Resta de objetos
#__mul__ : Multiplicación de objetos
#__getitem__ : Acceso a elementos mediante índices
#__setitem__ : Modificación de elementos mediante índices