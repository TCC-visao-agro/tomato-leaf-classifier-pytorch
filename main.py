import torch
import torchvision

torch.device("cuda")
def main():
    print(torch.version.cuda)


if __name__ == '__main__':
    main()
