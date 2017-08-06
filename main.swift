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
// FUNCTIONALITY
//--------------------------------------------------

// define a string tensor
let tensor = try TF.Tensor.Scalar("Hello, Perfect TensorFlow! 🇨🇳🇨🇦")

// declare a new graph
let g = try TF.Graph()

// turn the tensor into an operation
let op = try g.const(tensor: tensor, name: "hello")

// run a session
let o = try g.runner().fetch(op).addTarget(op).run()

// decode the result      
let decoded = try TF.Decode(strings: o[0].data, count: 1)

// check the result
let s2 = decoded[0].string
print(s2)
