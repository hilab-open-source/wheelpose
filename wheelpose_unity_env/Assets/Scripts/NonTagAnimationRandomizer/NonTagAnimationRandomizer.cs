using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

namespace Unity.CV.SyntheticHumans.Randomizers
{
    [Serializable]
    [AddRandomizerMenu("Custom/Non Tagged Animation Randomizer")]
    public class NonTagAnimationRandomizer : Randomizer
    {
        const string k_ClipName = "PlayerIdle";
        const string k_StateName = "Base Layer.RandomState";
        
        private FloatParameter animationTime = new FloatParameter { value = new UniformSampler(0f, 1f) };
        public CategoricalParameter<AnimationClip> animations;
        
        private void RandomizeAnimation(NonTagAnimationRandomizerTag tag)
        {
            if (!tag.gameObject.activeInHierarchy)
                return;

            var animator = tag.gameObject.GetComponent<Animator>();
            animator.applyRootMotion = tag.applyRootMotion;

            var overrider = tag.animatorOverrideController;
        
            if (overrider != null)
            {
                overrider[k_ClipName] = (AnimationClip) animations.Sample();
                animator.Play(k_StateName, 0, animationTime.Sample());

                // Unity won't update the animator until this frame is ready to render.
                // Force to update the animator and human poses in the same frame for the collision checking in the randomizers
                // The delta time must be greater than 0 to apply the root motion
                animator.Update(0.001f);
            }
        }

        protected override void OnIterationStart()
        {
            var taggedObjects = tagManager.Query<NonTagAnimationRandomizerTag>();
            foreach (var taggedObject in taggedObjects)
            {
                RandomizeAnimation(taggedObject);
            }
        }
    }
}
