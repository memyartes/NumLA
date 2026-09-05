from sklearn.datasets import load_breast_cancer


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(X.shape, y.shape)

    # your code here \/
    ...
    # your code here /\


if __name__ == "__main__":
    main()
