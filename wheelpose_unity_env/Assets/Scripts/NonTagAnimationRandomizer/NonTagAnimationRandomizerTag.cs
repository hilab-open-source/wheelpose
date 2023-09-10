using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

namespace Unity.CV.SyntheticHumans.Randomizers
{
    [RequireComponent(typeof(Animator))]
    public class NonTagAnimationRandomizerTag : RandomizerTag
    {
        public bool applyRootMotion = false;
        const string k_ClipName = "PlayerIdle";
        const string k_StateName = "Base Layer.RandomState";
        AnimatorOverrideController m_Controller;
        
        public AnimatorOverrideController animatorOverrideController
        {
            get
            {
                if (m_Controller == null)
                {
                    var animator = gameObject.GetComponent<Animator>();
                    var runtimeAnimatorController = Resources.Load<RuntimeAnimatorController>("AnimationRandomizerController");
                    m_Controller = new AnimatorOverrideController(runtimeAnimatorController);
                    animator.runtimeAnimatorController = m_Controller;
                }

                return m_Controller;
            }
        }
    }
}
