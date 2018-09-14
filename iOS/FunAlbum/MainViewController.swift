//
//  MainViewController.swift
//  FunAlbum
//
//  Created by Jinbin Xie on 13/9/18.
//  Copyright © 2018年 Jinbin Xie. All rights reserved.

import UIKit

class MainViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    @IBOutlet weak var imageView: UIImageView!
    
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
        let imgPath = Bundle.main.path(forResource: "test", ofType: "jpg")
        let newImage = UIImage(contentsOfFile: imgPath!)
        imageView.image = newImage
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destinationViewController.
        // Pass the selected object to the new view controller.
    }
    */

    // MARK: 用于弹出选择的对话框界面
    var selectorController: UIAlertController {
        let controller = UIAlertController(title: nil, message: nil, preferredStyle: .actionSheet)
        controller.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil)) // 取消按钮
        controller.addAction(UIAlertAction(title: "Take Photo", style: .default) { action in
            self.selectorSourceType(type: .camera)
        }) // 拍照选择
        controller.addAction(UIAlertAction(title: "Photo Library", style: .default) { action in
            self.selectorSourceType(type: .photoLibrary)
        }) // 相册选择
        return controller
    }
    
    @IBAction func onSelectFace(_ sender: Any) {
        present(selectorController, animated: true, completion: nil)
    }
    
    func selectorSourceType(type: UIImagePickerControllerSourceType) {
        imagePickerController.sourceType = type
        imagePickerController.delegate = self
        // 打开图片选择器
        present(imagePickerController, animated: true, completion: nil)
    }
    
    // MARK: 图片选择器界面
    lazy var imagePickerController: UIImagePickerController = {
        let imgPicker = UIImagePickerController()
        imgPicker.modalPresentationStyle = UIModalPresentationStyle.fullScreen
        imgPicker.allowsEditing = false
        imgPicker.delegate = self
        return imgPicker
    }()
    
    // MARK: 当图片选择器选择了一张图片之后回调
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        dismiss(animated: true, completion: nil)
        
        imageView.image = info[UIImagePickerControllerOriginalImage] as? UIImage // 显示图片
        imageView.contentMode = .scaleAspectFit // 缩放显示, 便于查看全部的图片
        
        NCNNWrapper.reSampleFace(imageView.image!)
    }
    
    // MARK: 当点击图片选择器中的取消按钮时回调
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil) // 效果一样的...
    }
}
