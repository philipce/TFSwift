//
//  main.swift
//  TFSwift
//
//  Created by Philip on 8/6/17.
//  Copyright © 2017 pce. All rights reserved.
//

//--------------------------------------------------
// IMPORTS
//--------------------------------------------------

import Foundation

// TensorFlowAPI contains most API functions defined in libtensorflow.so
import TensorFlowAPI

// This is the Swift version of TensorFlow classes and objects
import PerfectTensorFlow

// To keep the naming consistency with TensorFlow in other languages such as
// Python or Java, making an alias of `TensorFlow` Class is a good idea:
public typealias TF = TensorFlow

//--------------------------------------------------
// OPEN
//--------------------------------------------------

// Before using ANY ACTUAL FUNCTIONS of Perfect TensorFlow framework, TF.Open() must be called:

// this action will load all api functions defined
// in /usr/local/lib/libtensorflow.so
try TF.Open()

// Please also note that you can active the library with a specific path, alternatively, especially
// in case of different versions or CPU/GPU library adjustment required:
// this action will load the library with the path
// try TF.Open("/path/to/DLL/of/libtensorflow.so")

//--------------------------------------------------
// HELLO
//--------------------------------------------------

// define a string tensor
let tensor = try TF.Tensor.Scalar("Hello, Perfect TensorFlow! 🇨🇳🇨🇦")

// declare a new graph
var g = try TF.Graph()

// turn the tensor into an operation
let op = try g.const(tensor: tensor, name: "hello")

// run a session
var o = try g.runner().fetch(op).addTarget(op).run()

// decode the result      
let decoded = try TF.Decode(strings: o[0].data, count: 1)

// check the result
let s2 = decoded[0].string
print(s2)

//--------------------------------------------------
// MATRIX OPERATIONS
//--------------------------------------------------

/* Matrix Multiply:
| 1 2 |   |0 1|   |0 1|
| 3 4 | * |0 0| = |0 3|
*/
// input the matrix.
let tA = try TF.Tensor.Matrix([[1, 2], [3, 4]])
let tB = try TF.Tensor.Matrix([[0, 0], [1, 0]])

// adding tensors to graph 
g = try TF.Graph()
let A = try g.const(tensor: tA, name: "Const_0")
let B = try g.const(tensor: tB, name: "Const_1")

// define matrix multiply operation
let v = try g.matMul(l: A, r: B, name: "v", transposeB: true)

// run the session
o = try g.runner().fetch(v).addTarget(v).run()
let m:[Float] = try o[0].asArray()
print(m)
// m shall be [0, 1, 0, 3]

//--------------------------------------------------
// MODEL LOAD
//--------------------------------------------------

g = try TF.Graph()

// the meta signature info defined in a saved model
let metaBuf = try TF.Buffer()

// load the session
let session = try g.load(
	exportDir: "/path/to/saved/model",
	tags: ["tag1", "tag2"],
	metaGraphDef: metaBuf)




