package model

import "gonum.org/v1/gonum/mat"

type NetConfig struct {
	InNeurons     int
	OutNeurons    int
	HiddenNeurons int
	NumberEpochs  int
	LearningRate  float64
}

type Net struct {
	Cnf     NetConfig
	WHidden *mat.Dense
	BHidden *mat.Dense
	WOut    *mat.Dense
	BOut    *mat.Dense
}

func (n *Net) New(cnf NetConfig) {
	n.Cnf = cnf
}
