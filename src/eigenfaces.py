from sklearn.datasets import fetch_olivetti_faces

IMG = (64, 64)


def main():
    data = fetch_olivetti_faces(shuffle=False)
    X, y = data.data, data.target
    print(X.shape, y.shape)

    # your code here \/
    ...
    # your code here /\


if __name__ == "__main__":
    main()
