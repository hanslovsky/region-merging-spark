package de.hanslovsky.util;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;

import bdv.viewer.Source;
import bdv.viewer.ViewerPanel;
import bdv.viewer.state.SourceState;
import net.imglib2.RealRandomAccess;
import net.imglib2.ui.OverlayRenderer;

public class ValueDisplayListener implements MouseMotionListener, OverlayRenderer
{

	public static interface ToString< T >
	{
		String toString( T t );
	}

	private final HashMap< Source< ? >, RealRandomAccess< ? > > accesses;

	private final ViewerPanel viewer;

	private int width = 0;

	private int height = 0;

	private int x;

	private int y;

	private final HashMap< Source< ? >, Function< ?, String > > toString;

	public ValueDisplayListener( final HashMap< Source< ? >, RealRandomAccess< ? > > accesses, final ViewerPanel viewer, final HashMap< Source< ? >, Function< ?, String > > toString )
	{
		super();
		this.accesses = accesses;
		this.viewer = viewer;
		this.toString = toString;
		this.viewer.getDisplay().addOverlayRenderer( this );
		this.viewer.getDisplay().addMouseMotionListener( this );
	}

	public ValueDisplayListener( final HashMap< Source< ? >, RealRandomAccess< ? > > accesses, final ViewerPanel viewer )
	{
		this( accesses, viewer, new HashMap<>() );
	}

	public ValueDisplayListener( final ViewerPanel viewer )
	{
		this( new HashMap<>(), viewer, new HashMap<>() );
	}

	public < T > void addSource( final Source< ? > source, final RealRandomAccess< T > access, final Function< T, String > toString )
	{
		accesses.put( source, access );
		this.toString.put( source, toString );
	}

	@Override
	public void mouseDragged( final MouseEvent e )
	{}

	@Override
	public void mouseMoved( final MouseEvent e )
	{
		x = e.getX();
		y = e.getY();

		if ( viewer.isVisible() )
			viewer.getDisplay().repaint();
	}

	@Override
	public void drawOverlays( final Graphics g )
	{
		if ( viewer.isVisible() )
		{
			final Graphics2D g2d = ( Graphics2D ) g;

			g2d.setRenderingHint( RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON );
			g2d.setComposite( AlphaComposite.SrcOver );

			final int w = 176;
			final int h = 11;

			final int top = height - h;
			final int left = width - w;

			g2d.setColor( Color.white );
			g2d.fillRect( left, top, w, h );
			g2d.setColor( Color.BLACK );
			final List< SourceState< ? > > sources = viewer.getState().getSources();
			final Source< ? > source = sources.get( viewer.getState().getCurrentSource() ).getSpimSource();
			if ( accesses.containsKey( source ) )
			{
				final RealRandomAccess< ? > access = accesses.get( source );
				final Function ts = toString.containsKey( source ) ? toString.get( source ) : o -> o.toString();
//				viewer.getGlobalMouseCoordinates( access );
				final Object val = getVal( x, y, access, viewer );
				final String string = ( String ) ts.apply( val );
				g2d.drawString( string, left + 1, top + h - 1 );
//				drawBox( "selection", g2d, top, left, fid );
			}
		}
	}

	@Override
	public void setCanvasSize( final int width, final int height )
	{
		this.width = width;
		this.height = height;
	}

	public static < T > T getVal( final int x, final int y, final RealRandomAccess< T > access, final ViewerPanel viewer )
	{
		access.setPosition( x, 0 );
		access.setPosition( y, 1 );
		access.setPosition( 0, 2 );

		viewer.displayToGlobalCoordinates( access );

		return access.get();
	}

}