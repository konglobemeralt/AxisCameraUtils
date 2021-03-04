var u = 101;
var v = 101;

var scalingFactor = 100;

var Xcenter = 10.2;
var Ycenter = 9.1;
var Zcenter = 91.1;

var worldPoints = [[Xcenter,Ycenter,Zcenter],
		[11.0,5.0,91.0],
		[20.8,5.0,91.4],
		[11.0,13.5,91.4],
		[20.8,13.5,91.9]];

var imagePoints = [[307,227],
		[315,187],
		[425,187],
		[314,282],
		[426,282]];

var Xcenter = 10.2;
var Ycenter = 9.1;
var Zcenter = 91.1;

var worldPoints = [[Xcenter,Ycenter,Zcenter],
		[11.0,5.0,91.0],
		[20.8,5.0,91.4],
		[11.0,13.5,91.4],
		[20.8,13.5,91.9]];

var imagePoints = [[307,227],
		[315,187],
		[425,187],
		[314,282],
		[426,282]];

//Create uv vector
const uv = Math.Matrix([[u], [v], [1]]);

//scale UVvector
const scaledUV = Math.multiply(scalingFactor, uv);

//get A matrix
const Amatrix = Math.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
//invert a matrix
const invAmatrix = Math.matrix(Amatrix);
//invert a matrix
const dot = Math.dotMultiply(invAmatrix, scaledUV)

//Create tvec vector
const uv = Math.Matrix([[1], [v1], [1]]);


msg.payload = scaledUV;
return msg;
