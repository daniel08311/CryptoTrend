var http = require('http');
var fs = require('fs');
var express = require('express');
var path    = require("path");
var bodyParser = require('body-parser');

var app = express();

app.use(bodyParser.urlencoded({ extended: true }));

var data = JSON.parse(fs.readFileSync('predict_3-Hour.json', 'utf8'));

setInterval(function() {
 data = JSON.parse(fs.readFileSync('predict_3-Hour.json', 'utf8'));
}, 50000);

app.get('/', function (req, res) {
	res.sendFile(path.join(__dirname+'/index.html'));
	io.sockets.on('connection', function (socket) {
		setInterval(function() {
			//var data = JSON.parse(fs.readFileSync('predict_3-Hour.json', 'utf8'));
			socket.emit('exchange', {
				test : data
			});
			socket.emit('data', {
				test : data.binance
			})
		}, 3000);

	});
})


var server = app.listen("12345", "192.168.11.115", function () {
   var host = server.address().address
   var port = server.address().port
   console.log("Example app listening at http://%s:%s", host, port)
})


var io = require('socket.io').listen(server);
io.sockets.setMaxListeners(50);
// io.sockets.on('connection', function (socket) {
// 	setInterval(function() {
// 		socket.emit('data', {
// 			reward: data_dict.reward / 1000000000000000000,
// 			players: data_dict.players ,
// 			estTimestamp: data_dict.timestamp,
// 			bets: data_dict.betCount,
// 			price: data_dict.price / 1000000000000000000,
//       leader: data_dict.leader,
//       leader_2: data_dict.leader_2,
//       leader_3: data_dict.leader_3,
//       expireDate: data_dict.expireDate
// 		});
// 	}, 1000);
// });
