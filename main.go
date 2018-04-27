package main

import (
	"github.com/Rhymen/ml/neuronet"
	"fmt"
	"os"
	"time"
)

func main() {
	load := false

	now := time.Now()

	n := neuronet.New(784, 100, 10, 0.1)
	var err error

	if load {
		n, err = neuronet.FromCheckpoint("./checkpoints/784-100-10-01.gob")
	} else {
		err = n.TrainNetwork("./mnist_dataset/mnist_train_100.csv", 2)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading training data: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("finished training after %v\n", time.Since(now))

	now = time.Now()
	rate, err := n.TestNetwork("./mnist_dataset/mnist_test_10.csv")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading training data: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("finished testing after %v: %v\n", time.Since(now), rate)

	err = n.CreateCheckpoint("./checkpoints/784-100-10-01.gob")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating checkpoint: %v\n", err)
		os.Exit(1)
	}
}
