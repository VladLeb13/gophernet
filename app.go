package main

import (
	"log"
	"strconv"
	"strings"
	"time"

	classifier "github.com/VladLeb13/gophernet/run"
)

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)

	var (
		data   chan classifier.Data
		result chan []string
		status chan bool
	)

	data = make(chan classifier.Data)
	result = make(chan []string)
	status = make(chan bool)

	d := classifier.Data{}
	d.RawData = append(d.RawData, []string{"0.020488888", "0.213333333", "0.344444242", "0.4", "0.20", "0.20", "1.0", "0.0", "0.0"})
	d.RawData = append(d.RawData, []string{"0.081922222", "0.213333333", "0.344444242", "0.4", "0.20", "0.20", "0.0", "1.0", "0.0"})
	d.RawData = append(d.RawData, []string{"0.081922222", "0.213333333", "0.344444242", "0.4", "0.20", "0.20", "1.0", "0.0", "0.0"})

	go classifier.Classifier(data, result, status)

	time.Sleep(30 * time.Second)

	var init_manager int
	for init_manager != 1 {
		select {
		case ok := <-status:
			if ok {
				log.Println("Init success")
				init_manager = 1
				break
			}
		default:
			status <- false
			init_manager = 1
			return
		}
	}

	data <- d

	res := <-result

	answer := strings.Join(res, " ")
	for i, v := range d.RawData {
		elem := strings.Join(v, " ")
		if answer == elem {
			log.Println("bingo!!! утверждение номер: " + strconv.Itoa(i) + " верно")
			log.Println(v)
		}
	}

}
