# person.py

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def birthday(self):
        self.age += 1

    def __str__(self):
        return f"{self.name} ist {self.age} Jahre alt."

# Testen der Klasse
if __name__ == "__main__":
    person = Person("Max", 30)
    print(person)  # Max ist 30 Jahre alt.
    person.birthday()
    print(person)  # Max ist 31 Jahre alt.
