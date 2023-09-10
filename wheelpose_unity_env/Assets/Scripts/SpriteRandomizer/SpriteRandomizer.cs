using System;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers.Tags;
using UnityEngine.Scripting.APIUpdating;

namespace UnityEngine.Perception.Randomization.Randomizers
{
    [Serializable]
    [AddRandomizerMenu("Perception/Sprite Randomizer")]
    public class SpriteRandomizer : Randomizer
    {
        /// <summary>
        /// The list of sprites to sample and apply to target objects
        /// </summary>
        [Tooltip("The list of sprites to sample and apply to target objects.")]
        public CategoricalParameter<Texture2D> sprite;

        /// <summary>
        /// Randomizes the material texture of tagged objects at the start of each scenario iteration
        /// </summary>
        protected override void OnIterationStart()
        {
            var tags = tagManager.Query<SpriteRandomizerTag>();
            foreach (var tag in tags)
            {
                var spriteRenderer = tag.GetComponent<SpriteRenderer>();
                var tex = sprite.Sample();
                var s = Sprite.Create(tex, new Rect(0.0f, 0.0f, tex.width, tex.height), new Vector2(0.5f, 0.5f), 100.0f);
                spriteRenderer.sprite = s;
            }
        }
    }
}