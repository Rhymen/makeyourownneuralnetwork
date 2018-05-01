package main

import (
	"github.com/Rhymen/ml/neuronet"
	"fmt"
	"os"
	"time"
)

const (
	load           = false
	save           = false
	iNodes         = 784
	hNodes         = 100
	oNodes         = 10
	lr             = 0.1
	checkpointFile = "./checkpoints/784-100-10-01.gob"
	trainFile      = "./mnist_dataset/mnist_train_100.csv"
	epochs         = 2
	testFile       = "./mnist_dataset/mnist_test_10.csv"
)

func main() {
	now := time.Now()

	n := neuronet.New(iNodes, hNodes, oNodes, lr)
	var err error

	if load {
		n, err = neuronet.FromCheckpoint(checkpointFile)
	} else {
		err = n.TrainNetwork(trainFile, epochs)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading training data: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("finished training after %v\n", time.Since(now))

	now = time.Now()
	rate, err := n.TestNetwork(testFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading training data: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("finished testing after %v: %v\n", time.Since(now), rate)

	if save {
		err = n.CreateCheckpoint(checkpointFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error creating checkpoint: %v\n", err)
			os.Exit(1)
		}
	}
}
