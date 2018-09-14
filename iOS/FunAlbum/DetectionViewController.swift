//
//  DetectionViewController.swift
//  FunAlbum
//
//  Created by Jinbin Xie on 13/9/18.
//  Copyright © 2018年 Jinbin Xie. All rights reserved.
//

import UIKit
import AVFoundation

class DetectionViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    @IBOutlet weak var imageView: UIImageView!
    
    var session: AVCaptureSession!
    var device: AVCaptureDevice!
    var output: AVCaptureVideoDataOutput!
    var cameraPosition = AVCaptureDevice.Position.back
    
    private func prepareCamera() {
        // Prepare a video capturing session.
        self.session = AVCaptureSession()
        self.session.sessionPreset = AVCaptureSession.Preset.vga640x480 // not work in iOS simulator
        self.device = AVCaptureDevice.default(.builtInWideAngleCamera, for: AVMediaType.video, position: cameraPosition)
        if (self.device == nil) {
            print("no device")
            return
        }
        do {
            let input = try AVCaptureDeviceInput(device: self.device)
            self.session.addInput(input)
        } catch {
            print("no device input")
            return
        }
        self.output = AVCaptureVideoDataOutput()
        self.output.videoSettings = [ kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA) ]
        let queue: DispatchQueue = DispatchQueue(label: "videocapturequeue", attributes: [])
        self.output.setSampleBufferDelegate(self, queue: queue)
        self.output.alwaysDiscardsLateVideoFrames = true
        if self.session.canAddOutput(self.output) {
            self.session.addOutput(self.output)
        } else {
            print("could not add a session output")
            return
        }
        do {
            try self.device.lockForConfiguration()
            self.device.activeVideoMinFrameDuration = CMTimeMake(1, 20) // 20 fps
            self.device.unlockForConfiguration()
        } catch {
            print("could not configure a device")
            return
        }
    }
    override func viewDidLoad() {
        super.viewDidLoad()
        
        NCNNWrapper.initialize()
        
        prepareCamera()
        self.session.startRunning()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    override var shouldAutorotate : Bool {
        return false
    }
    
    @IBAction func onSwitchCamera(_ sender: Any) {
        self.session.stopRunning()
        
        if cameraPosition == AVCaptureDevice.Position.front {
            cameraPosition = AVCaptureDevice.Position.back
        } else {
            cameraPosition = AVCaptureDevice.Position.front
        }

        prepareCamera()
        self.session.startRunning()
    }
    
    func captureOutput(_ captureOutput: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        // Convert a captured image buffer to UIImage.
        guard let buffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("could not get a pixel buffer")
            return
        }
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags.readOnly)
        let image = CIImage(cvPixelBuffer: buffer).oriented(CGImagePropertyOrientation.right)
        let capturedImage = UIImage(ciImage: image)
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags.readOnly)
        
        // This is a filtering sample.
        let resultImage = NCNNWrapper.detectFace(capturedImage)
        
        // Show the result.
        DispatchQueue.main.async(execute: {
            self.imageView.image = resultImage
        })
    }
}
