import random
# zastosowana funkcja aktywacji to skok jednostkowy:
# 0 na wyjsciu -> dla wartosci wejsciowych mniejszych niz 0
# 1 na wyjsciu -> dla wartosci na wejsciu rownych badz wiekszych niz 0
def skok_jednostkowy(x):
    if x >= 0:
        return 1
    if x < 0:
        return 0


def uczenie_perceptronu(wyjscia_oczekiwane, epochs):
    print('**************************************** poczatek uczenia sieci ****************************************')
    # wartosc wspolczynnika alfa, decyduje on o tym jak duze beda zmiany wartosci wag w czasie uczenia:
    # duza wartosc wspolczynnika alfa -> duze zmiany wartosci
    # mala wartosc wspolczynnika alfa -> male zmiany wartosci
    alfa = 0.2
    # wagi poczatkowe - losowe liczby z przedzialu (-0.5, 0.5)
    wektor_wag = [random.random() - 0.5, random.random() - 0.5]
    # wartosc biasu - stala dzieki ktorej mozna regulowac czulosc perceptronu
    # mala wartosc progu - perceptron latwo pobudzic
    # duza wartosc progu - perceptron trudno pobudzic
    bias = 0.2
    # inicjalizacja wejsc jako wektorow dwu-elementowych
    wejscia = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    # zewnetrzna petla okresla ile razy na wejsciu perceptronu zostana podane wartosci wejsciowe
    for i in range(0, epochs):
        print('epoch:', i)
        for index, wektor_wejsc in enumerate(wejscia):
            # wartosc wyjscia obliczana jest jako suma iloczynow wejsc i odpowiadających im wag
            # od wartosci tej odejmowany jest bias - dzieki temu mozna regulowac wartosc graniczna,
            # dla ktorej perceptron zostanie pobudzony
            # na koniec obliczana jest funkjca aktywacji z wejscia - w tym przypadku skok jednostkowy
            wyjscie_rzeczywiste = skok_jednostkowy((wektor_wejsc[0] * wektor_wag[0] +
                                                    wektor_wejsc[1] * wektor_wag[1]) - bias)
            # obliczany jest blad jako roznica wartosci oczekiwanej i wartosci rzeczywistej
            blad = wyjscia_oczekiwane[index] - wyjscie_rzeczywiste
            # do odpowiednich wag dodawana jest poprawka -> odpowiednia wartosc wejscia mnozona przez wartosc bledu
            # oraz wspolczynnik alfa
            # dzieki temu, gdy:
            # wartosc wyjsciowa jest za duza -> wartosc wagi jest zmniejszana
            # gdy jest za mala -> waga jest zwiekszana
            # dodatkowo wraz ze wzrostem bledu wzrastac bedzie wartosc poprawki
            wektor_wag[0] = alfa * wektor_wejsc[0] * blad + wektor_wag[0]
            wektor_wag[1] = alfa * wektor_wejsc[1] * blad + wektor_wag[1]
            print('wartosc bledu:', blad, 'wektor wag: [', round(wektor_wag[0], 3), round(wektor_wag[1], 3), ']')
    return wektor_wag


def test_uczenia(wyjscie_oczekiwane, wektor_wag, bias):
    # funkcja sprawdza skutecznosc nauczonej sieci
    print('****************************************** test uczenia sieci ******************************************')
    wejscia = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for index, wektor_wejsc in enumerate(wejscia):
        wyjscie_rzeczywiste = skok_jednostkowy((wektor_wejsc[0] * wektor_wag[0] +
                                                    wektor_wejsc[1] * wektor_wag[1]) - bias)
        print(wektor_wejsc, '->', wyjscie_rzeczywiste, wyjscie_oczekiwane[index] == wyjscie_rzeczywiste)

# prblemy liniowo separowalne
# bramka AND
# 0|0->0
# 0|1->0
# 1|0->0
# 1|1->1
wagi_AND = uczenie_perceptronu(wyjscia_oczekiwane=[0, 0, 0, 1], epochs=10)
test_uczenia(wyjscie_oczekiwane=[0, 0, 0, 1],  wektor_wag=wagi_AND, bias=0.2)
# bramka OR
# 0|0->0
# 0|1->1
# 1|0->1
# 1|1->1
wagi_OR = uczenie_perceptronu(wyjscia_oczekiwane=[0, 1, 1, 1], epochs=10)
test_uczenia(wyjscie_oczekiwane=[0, 1, 1, 1], wektor_wag=wagi_OR, bias=0.2)

# problem ktory nie jest separowalny liniowo
# bramka XOR
# 0|0->0
# 0|1->1
# 1|0->1
# 1|1->0
wagi_XOR = uczenie_perceptronu(wyjscia_oczekiwane=[0, 1, 1, 0], epochs=10)
test_uczenia(wyjscie_oczekiwane=[0, 1, 1, 0], wektor_wag=wagi_XOR, bias=0.2)

# ************************************************* wnioski *************************************************
# na kolejnych etapach uczenia warotsc bledu malala dla problemow separowalnych liniowo -> bramka AND i OR
# oba problemy (AND i OR) sa liniowo separowalne, dzieki czemu mozna je rozwiazac za pomoca jednego perceptronu
# problem bramki XOR nalezy rozwiazac np za pomoca sieci wielowarstwowej, gdyz nie jest on liniowo separowalny
# uczenie perceptronu nalezy wykonywac do czasu, gdy wartosci wag sieci przestaja ulegac zmianom
# byl to przyklad uczenia z nauczycielem - oczekiwane wartosci wyjsciowe byly podawane sieci