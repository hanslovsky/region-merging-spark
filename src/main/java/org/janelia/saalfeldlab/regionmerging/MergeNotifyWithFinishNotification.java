package org.janelia.saalfeldlab.regionmerging;

import org.janelia.saalfeldlab.regionmerging.MergeNotify;

public interface MergeNotifyWithFinishNotification extends MergeNotify
{

	public void notifyDone();

}
