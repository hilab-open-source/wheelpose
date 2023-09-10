using System;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.DataModel;
using UnityEngine.Rendering;

[RequireComponent(typeof(PerceptionCamera))]
public class CustomCaptureSettingsLabeler : CameraLabeler
{
    //public GameObject targetLight;
    public GameObject targetCamera;

    //public GameObject target;
    public GameObject[] lightSources;

    public override string description => "Settings labeler";
    public override string labelerId => "Settings labeler";
    protected override bool supportsVisualization => false;

    MetricDefinition lightPositionMetricDefinition;
    MetricDefinition lightRotationMetricDefinition;
    MetricDefinition lightIntensityMetricDefinition;
    MetricDefinition lightColorMetricDefinition;

    MetricDefinition cameraPositionMetricDefinition;
    MetricDefinition cameraRotationMetricDefinition;
    MetricDefinition cameraFieldOfViewMetricDefinition;
    MetricDefinition cameraFocalLengthMetricDefinition;

    //AnnotationDefinition boundingBoxAnnotationDefinition;
    //SensorHandle cameraSensorHandle;

    protected override void Setup()
    {
        //Metrics and annotations are registered up-front
        lightPositionMetricDefinition = new MetricDefinition(
            "LightMetric",
            "LightPosition",
            "The World-space position of the light"
        );
        DatasetCapture.RegisterMetric(lightPositionMetricDefinition);

        lightRotationMetricDefinition = new MetricDefinition(
            "LightMetric",
            "LightRotation",
            "The world-space rotation of the light"
        );
        DatasetCapture.RegisterMetric(lightRotationMetricDefinition);

        lightIntensityMetricDefinition = new MetricDefinition(
            "LightMetric",
            "LightIntensity",
            "The intensity of the light"
        );
        DatasetCapture.RegisterMetric(lightIntensityMetricDefinition);

        lightColorMetricDefinition = new MetricDefinition(
            "LightMetric",
            "LightColor",
            "The color of the light"
        );
        DatasetCapture.RegisterMetric(lightColorMetricDefinition);

        cameraPositionMetricDefinition = new MetricDefinition(
            "CameraMetric",
            "CameraPosition",
            "The world-space position of the camera"
        );
        DatasetCapture.RegisterMetric(cameraPositionMetricDefinition);

        cameraRotationMetricDefinition = new MetricDefinition(
            "CameraMetric",
            "CameraRotation",
            "The world-space rotation of the camera"
        );
        DatasetCapture.RegisterMetric(cameraRotationMetricDefinition);

        cameraFieldOfViewMetricDefinition = new MetricDefinition(
            "CameraFOV",
            "42e7fa88-084b-423d-ba6e-830c711383e1",
            "The field of view of the camera"
        );
        DatasetCapture.RegisterMetric(cameraFieldOfViewMetricDefinition);

        cameraFocalLengthMetricDefinition = new MetricDefinition(
            "CameraFocalLength",
            "11aa1dfc-3495-467c-a998-71d9bfe6980e",
            "The focal length of the camera"
        );
        DatasetCapture.RegisterMetric(cameraFocalLengthMetricDefinition);
    }

    protected override void OnBeginRendering(ScriptableRenderContext scriptableRenderContext)
    {
        GenericMetric metric;
        var cameraFieldOfView = targetCamera.GetComponent<Camera>().fieldOfView;
        metric = new GenericMetric(new[] { cameraFieldOfView }, cameraFieldOfViewMetricDefinition,
            targetCamera.name);
        DatasetCapture.ReportMetric(cameraFieldOfViewMetricDefinition, metric);
        
        var cameraFocalLength = targetCamera.GetComponent<Camera>().focalLength;
        metric = new GenericMetric(new[] { cameraFocalLength }, cameraFocalLengthMetricDefinition,
            targetCamera.name);
        DatasetCapture.ReportMetric(cameraFocalLengthMetricDefinition, metric);
        // foreach (var lightObject in lightSources)
        // {
            // var lightName = lightObject.name;
            // Debug.Log(lightName);
            // //Report the light's position by manually creating the json array string.
            // var lightPosition = lightObject.transform.position;
            // var metric = new GenericMetric(new[] { lightPosition.x, lightPosition.y, lightPosition.z },
            //     lightPositionMetricDefinition, lightName, lightName);
            // DatasetCapture.ReportMetric(lightPositionMetricDefinition, metric);

            //     //Report the light's rotation by manually creating the json array string.
            //     var lightRotation = lightObject.transform.rotation.eulerAngles;
            //     DatasetCapture.ReportMetric(lightRotationMetricDefinition,
            //         $@"[{{ ""{"LightRotationX_" + lightName}"": {lightRotation.x},
            //                ""{"LightRotationY_" + lightName}"": {lightRotation.y},
            //                ""{"LightRotationZ_" + lightName}"": {lightRotation.z} }}]");
            //
            //     var lightIntensity = lightObject.GetComponent<Light>().intensity;
            //     DatasetCapture.ReportMetric(lightIntensityMetricDefinition,
            //         $@"[{{ ""{"LightIntensity_" + lightName}"": {lightIntensity} }}]");
            //
            //     var lightColor = lightObject.GetComponent<Light>().color;
            //     DatasetCapture.ReportMetric(lightColorMetricDefinition,
            //         $@"[{{ ""{"LightColorR_" + lightName}"": {lightColor.r},
            //                ""{"LightColorG_" + lightName}"": {lightColor.g},
            //                ""{"LightColorB_" + lightName}"": {lightColor.b},
            //                ""{"LightColorA_" + lightName}"": {lightColor.a} }}]");
            // }
            //
            // //Report the camera's position by manually creating the json array string.
            // var cameraPosition = targetCamera.transform.position;
            // DatasetCapture.ReportMetric(cameraPositionMetricDefinition,
            //     $@"[{{ ""CameraPositionX"": {cameraPosition.x}, ""CameraPositionY"": {cameraPosition.y}, ""CameraPositionZ"": {cameraPosition.z} }}]");
            //
            // //Report the camera's rotation by manually creating the json array string.q
            // var cameraRotation = targetCamera.transform.rotation.eulerAngles;
            // DatasetCapture.ReportMetric(cameraRotationMetricDefinition,
            //     $@"[{{ ""CameraRotationX"": {cameraRotation.x}, ""CameraRotationY"": {cameraRotation.y}, ""CameraRotationZ"": {cameraRotation.z} }}]");
            //
            //


            ////compute the location of the object in the camera's local space
            //Vector3 targetPos = transform.worldToLocalMatrix * target.transform.position;
            ////Report using the PerceptionCamera's SensorHandle if scheduled this frame
            //var sensorHandle = GetComponent<PerceptionCamera>().SensorHandle;
            //if (sensorHandle.ShouldCaptureThisFrame)
            //{
            //    sensorHandle.ReportAnnotationValues(
            //        boundingBoxAnnotationDefinition,
            //        new[] { targetPos });
            //}
        // }

    }
}