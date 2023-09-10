using System;
using UnityEngine;

namespace DefaultNamespace
{
    public class TransformToSyntheticHuman : MonoBehaviour
    {
        // for use in prefabs for synthetic humans
        // set the parent transform to the desired body part

        public string hookBodyPart;
        
        private GameObject hookObject;
        private GameObject parent;

        private GameObject leftHip;
        private GameObject leftKnee;

        private void Start()
        {
            // sets the position to a hook
            parent = transform.parent.gameObject;
            hookObject = GetDescendantWithName(parent, hookBodyPart);


            if (hookObject != null)
                transform.parent = hookObject.transform;
        }

        public GameObject GetDescendantWithName(GameObject obj, string descendantName)
        {
            GameObject descendant = null;
            foreach (Transform child in obj.transform)
            {
                if (child.name == descendantName)
                {
                    return child.gameObject;
                }

                descendant = GetDescendantWithName(child.gameObject, descendantName);
                if (descendant != null)
                {
                    return descendant;
                }
            }

            return descendant;
        }
    }
}