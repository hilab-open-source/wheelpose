using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace DefaultNamespace
{
    [ExecuteInEditMode]
    public class FBXToAnim : MonoBehaviour
    {
        [Tooltip("Relative to project directory")]
        public string inPath;
        [Tooltip("Relative to project directory")]
        public string outPath;

        private void Start()
        {
            ExtractAllAnims();
        }

        public void ExtractAllAnims()
        {
            inPath = inPath.Replace("\\", "/");
            outPath = outPath.Replace("\\", "/");
            
            foreach (var file in System.IO.Directory.GetFiles(Path.Join(System.IO.Directory.GetCurrentDirectory(), inPath), "*.fbx"))
            {
                Debug.Log(file);
    
                var fileName = Path.GetFileNameWithoutExtension(file);
                var filePath = $"{inPath}/{fileName}.fbx";
                var src = AssetDatabase.LoadAssetAtPath<AnimationClip>(filePath);
                var temp = new AnimationClip();
                EditorUtility.CopySerialized(src, temp);
                var savePath = $"{outPath}/{fileName}.anim";
                AssetDatabase.CreateAsset(temp, savePath);


            }
            AssetDatabase.SaveAssets();
        }
    }
}