# # S-SGD, no momentum
# python main.py --lr 0.001 --momentum 0.0 --model resnet --dataset cifar10 --epochs 100  --st 0 -s 1 --gpu-num 4 --port 6632 --cluster-data
# # Local-SGD, no momentum
# python main.py --lr 0.001 --momentum 0.0 --model resnet --dataset cifar10 --epochs 100  --st 0 -s 1 --gpu-num 4 --port 6633 --cluster-data --local --period 5
# # VRL-SGD, no momentum
# python main.py --lr 0.001 --momentum 0.0 --model resnet --dataset cifar10 --epochs 100  --st 0 -s 1 --gpu-num 4 --port 6634 --cluster-data --local --period 5 --vrl
# # proposed, no momentum
# python main.py --lr 0.001 --momentum 0.0 --model resnet --dataset cifar10 --epochs 100  --st 0 -s 1 --gpu-num 4 --port 6634 --cluster-data --local --period 5 --vrl --ghc

# # S-SGD, momentum
# python main.py --lr 0.001 --momentum 0.9 --model resnet --dataset cifar10 --epochs 100  --st 0 -s 1 --gpu-num 4 --port 6632 --cluster-data
# # Local-SGD, momentum
# python main.py --lr 0.001 --momentum 0.9 --model resnet --dataset cifar10 --epochs 100  --st 0 -s 1 --gpu-num 4 --port 6633 --cluster-data --local --period 5
# # VRL-SGD, momentum
# python main.py --lr 0.001 --momentum 0.9 --model resnet --dataset cifar10 --epochs 100  --st 0 -s 1 --gpu-num 4 --port 6634 --cluster-data --local --period 5 --vrl
# # proposed, no momentum
# python main.py --lr 0.001 --momentum 0.9 --model resnet --dataset cifar10 --epochs 100  --st 0 -s 1 --gpu-num 4 --port 6634 --cluster-data --local --period 5 --vrl --ghc

# # cgcn
# python main.py --lr 0.01 --momentum 0.0 --model cgcn --dataset graphData --epochs 100 --st 0 -s 1 --gpu-num 4 --port 6632 --cluster-data

# python main.py --lr 0.01 --momentum 0.0 --model cgcn --dataset graphData --epochs 100 --st 0 -s 1 --gpu-num 4 --port 6633 --cluster-data --local --period 5

# python main.py --lr 0.01 --momentum 0.0 --model cgcn --dataset graphData --epochs 100 --st 0 -s 1 --gpu-num 4 --port 6634 --cluster-data --local --period 5 --vrl

# python main.py --lr 0.01 --momentum 0.0 --model cgcn --dataset graphData --epochs 100 --st 0 -s 1 --gpu-num 4 --port 6634 --cluster-data --local --period 5 --vrl --ghc


# # cgcn
# python main.py --lr 0.01 --momentum 0.9 --model cgcn --dataset graphData --epochs 100 --st 0 -s 1 --gpu-num 4 --port 6632 --cluster-data

# python main.py --lr 0.01 --momentum 0.9 --model cgcn --dataset graphData --epochs 100 --st 0 -s 1 --gpu-num 4 --port 6633 --cluster-data --local --period 5

# python main.py --lr 0.01 --momentum 0.9 --model cgcn --dataset graphData --epochs 100 --st 0 -s 1 --gpu-num 4 --port 6634 --cluster-data --local --period 5 --vrl

# python main.py --lr 0.01 --momentum 0.9 --model cgcn --dataset graphData --epochs 100 --st 0 -s 1 --gpu-num 4 --port 6634 --cluster-data --local --period 5 --vrl --ghc

# cluster gcn
python ghcMain.py --dropout 0.2 --lr 0.02 --momentum 0.0 --model clustergcn --dataset ppi --epochs 300 --st 0 -s 1 --gpu-num 4 --port 6632 --cluster-data --diag_lambda 1
python ghcMain.py --dropout 0.2 --lr 0.02 --momentum 0.0 --model clustergcn --dataset ppi --epochs 300 --st 0 -s 1 --gpu-num 4 --port 6632 --cluster-data --diag_lambda 1 --local --period 5
python ghcMain.py --dropout 0.2 --lr 0.02 --momentum 0.0 --model clustergcn --dataset ppi --epochs 300 --st 0 -s 1 --gpu-num 4 --port 6632 --cluster-data --diag_lambda 1 --local --period 5 --vrl
python ghcMain.py --dropout 0.2 --lr 0.02 --momentum 0.0 --model clustergcn --dataset ppi --epochs 300 --st 0 -s 1 --gpu-num 4 --port 6632 --cluster-data --diag_lambda 1 --local --period 5 --vrl --ghc

python ghcMain.py --dropout 0.2 --lr 0.02 --momentum 0.9 --model clustergcn --dataset ppi --epochs 300 --st 0 -s 1 --gpu-num 4 --port 6632 --cluster-data --diag_lambda 1
python ghcMain.py --dropout 0.2 --lr 0.02 --momentum 0.9 --model clustergcn --dataset ppi --epochs 300 --st 0 -s 1 --gpu-num 4 --port 6632 --cluster-data --diag_lambda 1 --local --period 5
python ghcMain.py --dropout 0.2 --lr 0.02 --momentum 0.9 --model clustergcn --dataset ppi --epochs 300 --st 0 -s 1 --gpu-num 4 --port 6632 --cluster-data --diag_lambda 1 --local --period 5 --vrl
python ghcMain.py --dropout 0.2 --lr 0.02 --momentum 0.9 --model clustergcn --dataset ppi --epochs 300 --st 0 -s 1 --gpu-num 4 --port 6632 --cluster-data --diag_lambda 1 --local --period 5 --vrl --ghc

