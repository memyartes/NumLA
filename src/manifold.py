from sklearn.datasets import load_digits


def main():
    digits = load_digits()
    X, y = digits.data, digits.target
    print(X.shape, y.shape)

    # your code here \/
    ...
    # your code here /\


if __name__ == "__main__":
    main()
