using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

namespace DefaultNamespace
{
    public class AnimationStepVisualizer : MonoBehaviour
    {
        AnimatorOverrideController m_Controller;

        public GameObject model;
        public AnimationClip animation;
        public int nSteps = 4;
        public float animationSpeed = .1f;
        
        public Vector3 direction = new Vector3(1, 0, .5f);
        public float gap = 1f;
        public Vector3 rotation = new Vector3();
        
        public bool applyRootMotion = false;

        public bool pauseOnUpdate = false;

        public int visualizeIdx = -1;
        private List<GameObject> models = new List<GameObject>();

        public bool iterateIdx = false;

        private int currFrame;
        
        const string k_ClipName = "PlayerIdle";
        const string k_StateName = "Base Layer.RandomState";
        private void Start()
        {

            for (var i = 0; i < nSteps; i++)
            {
                var curr_model = Instantiate(model);
                
                // setting up the position
                curr_model.transform.parent = transform;
                curr_model.transform.localPosition = direction.normalized * (i * gap);
                curr_model.transform.localEulerAngles = rotation;
                // setting up the animator override
                var animator = curr_model.GetComponent<Animator>();
                var runtimeAnimatorController = Resources.Load<RuntimeAnimatorController>("AnimationRandomizerController");
                var overrider = new AnimatorOverrideController(runtimeAnimatorController);
                animator.runtimeAnimatorController = overrider;
                
                animator.speed = animationSpeed;
                animator.applyRootMotion = applyRootMotion;
                
                // setting up the animation
                overrider[k_ClipName] = animation;

                var normed_time = (float)i / ((float)nSteps - 1) - .00001f;
                animator.Play(k_StateName, 0, normed_time);
                
                // Unity won't update the animator until this frame is ready to render.
                // Force to update the animator and human poses in the same frame for the collision checking in the randomizers
                // The delta time must be greater than 0 to apply the root motion
                animator.Update(0.001f);
                models.Add(curr_model);
            }

        }

        private void Update()
        {
            if (iterateIdx)
                visualizeIdx = currFrame % nSteps;
            else
                visualizeIdx = -1;
            
            
            // reupdating the position every step
            for (var i = 0; i < nSteps; i++)
            {
                
                var curr_model = models[i];
                curr_model.transform.localEulerAngles = Vector3.zero;
                curr_model.transform.localPosition = direction.normalized * (i * gap);
                curr_model.transform.localEulerAngles = rotation;
                
                // focusing on one particular index if needed
                if (visualizeIdx == -1)
                    ToggleRenderers(curr_model, true);
                else if (visualizeIdx >= 0 && visualizeIdx < nSteps)
                {
                    if (i == visualizeIdx)
                        ToggleRenderers(curr_model, true);
                    else
                        ToggleRenderers(curr_model, false);
                }
            }

            if (pauseOnUpdate)
                Debug.Break();
            
            currFrame++;
        }
        
        private void ToggleRenderers(GameObject go, bool state)
        {
            var renderers = go.GetComponentsInChildren<Renderer>();
            for (var i = 0; i < renderers.Length; i++)
            {
                renderers[i].enabled = state;
            }
        }
        
    }
}