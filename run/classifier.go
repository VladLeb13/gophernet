package run

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"github.com/VladLeb13/gophernet/model"
	"github.com/VladLeb13/gophernet/tools"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type (
	Data struct {
		DataIn  *mat.Dense
		Labels  *mat.Dense
		RawData [][]string
	}
)

func (d *Data) Build() {
	inputsData := make([]float64, 6*len(d.RawData))
	labelsData := make([]float64, 3*len(d.RawData))

	// Will track the current index of matrix values.
	var inputsIndex int
	var labelsIndex int

	for _, record := range d.RawData {

		// Loop over the float columns.
		for i, val := range record {

			// Convert the value to a float.
			parsedVal, err := strconv.ParseFloat(string(val), 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i == 6 || i == 7 || i == 8 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}
	d.DataIn = mat.NewDense(len(d.RawData), 6, inputsData)
	d.Labels = mat.NewDense(len(d.RawData), 3, labelsData)

}
func Classifier(data chan Data, result chan []string, status chan bool) (err error) {
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	config := model.NetConfig{
		InNeurons:     6,
		OutNeurons:    3,
		HiddenNeurons: 5,
		NumberEpochs:  20000,
		LearningRate:  0.2,
	}

	network := &model.Net{}
	network.New(config)

	var accuracy float64
	for accuracy < 0.90 {
		select {
		case next := <-status:
			if !next {
				return errors.New("Init Fail")
			}
		default:
			accuracy = trainModel(network)
		}

	}

	status <- true

	for {
		select {
		case d := <-data:
			d.Build()
			predictions, err := network.Predict(d.DataIn)
			if err != nil {
				log.Fatal(err)
			}

			numPreds, _ := predictions.Dims()
			for i := 0; i < numPreds; i++ {

				// Get the label.
				labelRow := mat.Row(nil, i, d.Labels)
				var prediction int
				for idx, label := range labelRow {
					if label == 1.0 {
						prediction = idx
						break
					}
				}
				p := predictions.At(i, prediction)
				p1 := floats.Max(mat.Row(nil, i, predictions))

				if p == p1 {
					result <- d.RawData[i]
				}
			}
		default:
			time.Sleep(1 * time.Second)
		}
	}

}
func trainModel(network *model.Net) (accuracy float64) {
	_, filename, _, _ := runtime.Caller(0)

	dirPath := filepath.Dir(path.Dir(filename))

	inputs, labels := tools.MakeInputsAndLabels(dirPath + string(os.PathSeparator) + "data" + string(os.PathSeparator) + "train.csv")
	// Define our network architecture and learning parameters.
	if err := network.Train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	// Form the testing matrices.
	testInputs, testLabels := tools.MakeInputsAndLabels(dirPath + string(os.PathSeparator) + "data" + string(os.PathSeparator) + "test.csv")

	// Make the predictions using the trained model.
	predictions, err := network.Predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate the accuracy of our model.
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {

		// Get the label.
		labelRow := mat.Row(nil, i, testLabels)
		var prediction int
		for idx, label := range labelRow {
			if label == 1.0 {
				prediction = idx
				break
			}
		}

		// Accumulate the true positive/negative count.
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Calculate the accuracy (subset accuracy).
	accuracy = float64(truePosNeg) / float64(numPreds)

	// Output the Accuracy value to standard out.
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)

	return
}
