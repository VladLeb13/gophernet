package model

import (
	"errors"
	"math/rand"
	"time"

	"github.com/VladLeb13/gophernet/tools"
	"gonum.org/v1/gonum/mat"
)

// train обучает нейронную сеть, используя обратное распространение.
func (n *Net) Train(x, y *mat.Dense) error {
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHidden := mat.NewDense(n.Cnf.InNeurons, n.Cnf.HiddenNeurons, nil)
	bHidden := mat.NewDense(1, n.Cnf.HiddenNeurons, nil)
	wOut := mat.NewDense(n.Cnf.HiddenNeurons, n.Cnf.OutNeurons, nil)
	bOut := mat.NewDense(1, n.Cnf.OutNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Use backpropagation to adjust the weights and biases.
	if err := n.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// Define our trained neural network.
	n.WHidden = wHidden
	n.BHidden = bHidden
	n.WOut = wOut
	n.BOut = bOut

	return nil
}

// backpropagate завершает метод прямого распространения.
func (n *Net) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	for i := 0; i < n.Cnf.NumberEpochs; i++ {

		// Complete the feed forward process.
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 { return tools.Sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Complete the backpropagation.
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return tools.SigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Adjust the parameters.
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(n.Cnf.LearningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := tools.SumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(n.Cnf.LearningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(n.Cnf.LearningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := tools.SumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(n.Cnf.LearningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}

	return nil
}

// predict делает предсказание с помощью
func (n *Net) Predict(x *mat.Dense) (*mat.Dense, error) {

	// Check to make sure that our neuralNet value
	// represents a trained model.
	if n.WHidden == nil || n.WOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if n.BHidden == nil || n.BOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Complete the feed forward process.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, n.WHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + n.BHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return tools.Sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, n.WOut)
	addBOut := func(_, col int, v float64) float64 { return v + n.BOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}
