## build docker image

```
cd /-path-/code
docker build .
```

## run docker image

```
docker run --ipc=host -it finalsub
```

## Training code
```
./train.sh <path for training.csv> 
```
example :
```
./train.sh data/training.csv
```

## Testing code
```
./test.sh <path for training.csv> <path for testing.csv> <path for model (Default is 'model')> <path for submission.csv>
```
example:
```
./test.sh data/training.csv data/testing.csv model submission.csv
```